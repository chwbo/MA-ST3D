import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_gm_utils
import pickle as pkl
import re
from multiprocessing import Manager

from scipy.spatial.transform import Rotation as Rot
import pickle
import time

PSEUDO_LABELS = {}
GLBMEM_LABELS = {}
LOCMEM_LABELS = {}
RAW_PSEUDO_LABELS = {}
CUR_PSEUDO_LABELS = {}
MERGE_PARS = {}

def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            LOCMEM_LABELS.update(PSEUDO_LABELS)
            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    ps_glb_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_glb_label_e*.pkl'))
    if len(ps_label_list) == 0 or len(ps_glb_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    ps_glb_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1
        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            LOCMEM_LABELS.update(PSEUDO_LABELS)

        cur_glb_pkl = ps_glb_label_list[0]
        num_epoch = re.findall('ps_glb_label_e(.*).pkl', cur_glb_pkl)
        assert len(num_epoch) == 1
        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_glb_ps_label = pkl.load(open(cur_glb_pkl, 'rb'))
            GLBMEM_LABELS.update(latest_glb_ps_label)

        return cur_pkl

    return None


def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    print('Notice: for compute time of pseudo label generation')

    loc_ps_pos_meter = common_utils.AverageMeter()
    loc_ps_ign_meter = common_utils.AverageMeter()

    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)
    model.eval()
    # local memory is updated by new pseudo labels
    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        # generate local pseudo labels

        loc_ps_pos, loc_ps_ign = local_pathway_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            locmem_update=(cfg.SELF_TRAIN.get('LOCMEM_ENSEMBLE', None) and
                           cfg.SELF_TRAIN.LOCMEM_ENSEMBLE.ENABLED and
                           cur_epoch > 0)
        )
        loc_ps_pos_meter.update(loc_ps_pos)
        loc_ps_ign_meter.update(loc_ps_ign)

        if rank == 0:
            pbar.update()
            pbar.refresh()
    if rank == 0:
        pbar.close()

    # update global pseudo labels
    commu_utils.synchronize()
    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(RAW_PSEUDO_LABELS)
        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)
        RAW_PSEUDO_LABELS.update(new_pseudo_label_dict)

        part_locmem_label_list = commu_utils.all_gather(LOCMEM_LABELS)
        new_locmem_label_dict = {}
        for merge_labels in part_locmem_label_list:
            new_locmem_label_dict.update(merge_labels)
        LOCMEM_LABELS.update(new_locmem_label_dict)

        part_merge_pars_list = commu_utils.all_gather(MERGE_PARS)
        new_merge_pars_dict = {}
        for merge_labels in part_merge_pars_list:
            new_merge_pars_dict.update(merge_labels)
        MERGE_PARS.update(new_merge_pars_dict)

    ps_meters = {}

    if cfg.SELF_TRAIN.TRANS_MERGE:
        ps_meters = global_pathway_pseudo_label(rank, cur_epoch)

    else:
        if not cfg.SELF_TRAIN.LOCMEM_ENSEMBLE.ENABLED or cur_epoch == 0:
            CUR_PSEUDO_LABELS.update(RAW_PSEUDO_LABELS)
        else:
            CUR_PSEUDO_LABELS.update(LOCMEM_LABELS)

    ps_meters.update(
        {'avg_loc_ps_pos_meter': loc_ps_pos_meter.avg,
         'avg_loc_ps_ign_meter': loc_ps_ign_meter.avg})

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)

    return ps_meters

#from pcdet.datasets.kitti_odom.kitti_utils import read_calib as kitti_odom_read_calib

#from pcdet.datasets.kitti_odom.kitti_utils import read_poses as kitti_odom_read_poses

#def read_pose(sequ):
#
#    if cfg.DATA_CONFIG.DATASET == 'KittiOdomDataset':
#        poses = kitti_odom_read_poses(sequ, cfg.ROOT_DIR)
#
#    return poses
#
#def read_calib(sequ):
#
#    if cfg.DATA_CONFIG.DATASET == 'KittiOdomDataset':
#        calib = kitti_odom_read_calib(sequ, cfg.ROOT_DIR)
#
#    return calib

from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu, nms_normal_gpu, boxes_iou3d_gpu, boxes_iou_bev

def write_ply(pc, pc_file):
    float_formatter = lambda x: "%.4f" % x
    points =[]
    for l in range(pc.shape[0]):
        i = pc[l]
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                       int(i[3]), int(i[4]), int(i[5])))

    file = open(pc_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

from pcdet.utils import common_utils

def warp_loc_to_global(lidar_boxes, cam0_to_cam0world, tr_lidar_to_cam0):

    if tr_lidar_to_cam0.shape[0] < 4:
        tr_lidar_to_cam0_homo = np.identity(4)
        tr_lidar_to_cam0_homo[:3, :] = tr_lidar_to_cam0
    else:
        tr_lidar_to_cam0_homo = tr_lidar_to_cam0

    if cam0_to_cam0world.shape[0] < 4:
        cam0_to_cam0world_homo = np.identity(4)
        cam0_to_cam0world_homo[:3, :] = cam0_to_cam0world
    else:
        cam0_to_cam0world_homo = cam0_to_cam0world

    lidar_to_cam0world = np.matmul(cam0_to_cam0world_homo, tr_lidar_to_cam0_homo)
    tr_cam0_to_lidar_homo = np.linalg.inv(tr_lidar_to_cam0_homo)
    lidar_to_lidarworld = np.matmul(tr_cam0_to_lidar_homo, lidar_to_cam0world)

    # warp position
    lidar_centers = lidar_boxes[:, :3]
    lidar_centers_homo = np.ones([*lidar_centers.shape[:-1], lidar_centers.shape[-1] + 1], dtype=lidar_centers.dtype)
    lidar_centers_homo[:, :3] = lidar_centers
    lidarworld_centers = np.matmul(lidar_to_lidarworld, lidar_centers_homo.transpose()).transpose()
    lidarworld_centers = lidarworld_centers[:, :3]

    # warp orientation
    rot = Rot.from_matrix(lidar_to_lidarworld[:3, :3])
    theta = rot.as_euler('xyz')
    lidarworld_orient = lidar_boxes[:, 6] + theta[2]
    lidarworld_orient[lidarworld_orient > np.pi] -= 2.0*np.pi
    lidarworld_orient[lidarworld_orient < -1.0*np.pi] += 2.0 * np.pi

    lidarworld_boxes = np.zeros(lidar_boxes.shape)
    lidarworld_boxes[:, :3] = lidarworld_centers
    lidarworld_boxes[:, 3:6] = lidar_boxes[:, 3:6]
    lidarworld_boxes[:, 6] = lidarworld_orient
    lidarworld_boxes[:, 7:] = lidar_boxes[:, 7:]
    return lidarworld_boxes


def warp_global_to_loc(lidarworld_boxes, cam0_to_cam0world, tr_lidar_to_cam0):

    if tr_lidar_to_cam0.shape[0] < 4:
        tr_lidar_to_cam0_homo = np.identity(4)
        tr_lidar_to_cam0_homo[:3, :] = tr_lidar_to_cam0
    else:
        tr_lidar_to_cam0_homo = tr_lidar_to_cam0

    if cam0_to_cam0world.shape[0] < 4:
        cam0_to_cam0world_homo = np.identity(4)
        cam0_to_cam0world_homo[:3, :] = cam0_to_cam0world
    else:
        cam0_to_cam0world_homo = cam0_to_cam0world

    lidar_to_cam0world = np.matmul(cam0_to_cam0world_homo, tr_lidar_to_cam0_homo)
    tr_cam0_to_lidar_homo = np.linalg.inv(tr_lidar_to_cam0_homo)
    lidar_to_lidarworld = np.matmul(tr_cam0_to_lidar_homo, lidar_to_cam0world)
    lidarworld_to_lidar = np.linalg.inv(lidar_to_lidarworld)

    # prepare lidar centers
    lidarworld_centers = lidarworld_boxes[:, :3]
    lidarworld_centers_homo = np.ones([*lidarworld_centers.shape[:-1], lidarworld_centers.shape[-1] + 1], dtype=lidarworld_centers.dtype)
    lidarworld_centers_homo[:, :3] = lidarworld_centers
    lidar_centers_homo = np.matmul(lidarworld_to_lidar, lidarworld_centers_homo.transpose()).transpose()
    lidar_centers = lidar_centers_homo[:, :3]

    # warp orientation
    rot = Rot.from_matrix(lidarworld_to_lidar[:3, :3])
    theta = rot.as_euler('xyz')
    lidar_orient = lidarworld_boxes[:, 6] + theta[2]
    lidar_orient[lidar_orient > np.pi] -= 2.0 * np.pi
    lidar_orient[lidar_orient < -1.0 * np.pi] += 2.0 * np.pi

    lidar_boxes = np.zeros(lidarworld_boxes.shape)
    lidar_boxes[:, :3] = lidar_centers
    lidar_boxes[:, 3:6] = lidarworld_boxes[:, 3:6]
    lidar_boxes[:, 6] = lidar_orient
    lidar_boxes[:, 7:] = lidarworld_boxes[:, 7:]
    return lidar_boxes

def warp_points_loc_to_global(lidar_points, cam0_to_cam0world, tr_lidar_to_cam0):

    if tr_lidar_to_cam0.shape[0] < 4:
        tr_lidar_to_cam0_homo = np.identity(4)
        tr_lidar_to_cam0_homo[:3, :] = tr_lidar_to_cam0
    else:
        tr_lidar_to_cam0_homo = tr_lidar_to_cam0

    if cam0_to_cam0world.shape[0] < 4:
        cam0_to_cam0world_homo = np.identity(4)
        cam0_to_cam0world_homo[:3, :] = cam0_to_cam0world
    else:
        cam0_to_cam0world_homo = cam0_to_cam0world

    lidar_to_cam0world = np.matmul(cam0_to_cam0world_homo, tr_lidar_to_cam0_homo)
    tr_cam0_to_lidar_homo = np.linalg.inv(tr_lidar_to_cam0_homo)
    lidar_to_lidarworld = np.matmul(tr_cam0_to_lidar_homo, lidar_to_cam0world)

    # warp position
    lidar_points = lidar_points[:, :3]
    lidar_points_homo = np.ones([*lidar_points.shape[:-1], lidar_points.shape[-1] + 1], dtype=lidar_points.dtype)
    lidar_points_homo[:, :3] = lidar_points

    lidarworld_points_homo = np.matmul(lidar_to_lidarworld, lidar_points_homo.transpose()).transpose()
    lidarworld_points = lidarworld_points_homo[:, :3]

    return lidarworld_points

def verify_labels_iou3d(warped_boxes, merged_boxes, merge_iou='3D', verify_thr=0.7):
    gpu_warped_boxes = torch.from_numpy(warped_boxes).to(torch.float32).cuda()
    gpu_merged_boxes = torch.from_numpy(merged_boxes).to(torch.float32).cuda()

    if merge_iou == 'BEV':
        iou = boxes_iou_bev(gpu_warped_boxes, gpu_merged_boxes)
    else:
        iou = boxes_iou3d_gpu(gpu_warped_boxes, gpu_merged_boxes)
    max_warped_to_merged = iou.max(dim=-1)
    idx_verify_warped = torch.where(max_warped_to_merged[0] > verify_thr)[0]
    idx_verify_merged = max_warped_to_merged[1][idx_verify_warped]
    idx_verify_warped = idx_verify_warped.cpu().numpy()
    idx_verify_merged = idx_verify_merged.cpu().numpy()
    return idx_verify_warped, idx_verify_merged

def merge_with_max_boxes_max_scores(boxes, scores, merge_iou='3D', merge_cnt=3, merge_iou_thr=0.5, merge_minimum_score=0.5):
    gpu_boxes = torch.from_numpy(boxes).to(torch.float32).cuda()
    gpu_scores = torch.from_numpy(scores).to(torch.float32).cuda()

    gpu_nms_boxes = nms_gpu(gpu_boxes, gpu_scores, thresh=0.1)
    nms_valid = torch.zeros(gpu_boxes.shape[0]).bool()
    nms_valid[gpu_nms_boxes[0]] = True

    if merge_iou == 'BEV':
        boxes_iou = boxes_iou_bev(gpu_boxes, gpu_boxes).cpu()
    else:
        boxes_iou = boxes_iou3d_gpu(gpu_boxes, gpu_boxes).cpu()

    boxes_iou_over_thr = (boxes_iou >= merge_iou_thr)
    boxes_iou_over_thr_cnt = boxes_iou_over_thr.sum(dim=0)
    boxes_iou_over_cnt_valid = boxes_iou_over_thr_cnt >= merge_cnt

    merge_boxes_valid = nms_valid * boxes_iou_over_cnt_valid
    merged_boxes_valid_idx = torch.where(merge_boxes_valid)[0]

    merged_boxes = []
    merged_scores = []
    ## no merged box, return
    if len(merged_boxes_valid_idx) < 1:
        return None, None, merged_boxes_valid_idx

    valid_clusters = {}
    cluster_cnt = 0
    for valid in merged_boxes_valid_idx:
        valid_cluster_idx = torch.where(boxes_iou_over_thr[valid])
        valid_cluster = {cluster_cnt: valid_cluster_idx}
        valid_clusters.update(valid_cluster)
        cluster_cnt += 1

    for ik in valid_clusters.keys():
        cluster_idx = valid_clusters[ik]
        cluster_boxes = boxes[cluster_idx]
        cluster_scores = scores[cluster_idx]
        cluster_sort_idx = cluster_scores.argsort()
        cluster_sort_idx = cluster_sort_idx[::-1]
        cluster_boxes = cluster_boxes[cluster_sort_idx]
        cluster_scores = cluster_scores[cluster_sort_idx]
        ## remove angle diff over thr
        max_scored_angle = cluster_boxes[0, 6]
        cluster_angle_diff = np.abs(cluster_boxes[:, 6] - max_scored_angle)
        valid_angle_idx = cluster_angle_diff < np.pi / 12.0
        cluster_boxes = cluster_boxes[valid_angle_idx]
        cluster_scores = cluster_scores[valid_angle_idx]

        merged_boxes.append(cluster_boxes[0])
        # setting the minimum score for merged boxes
        cluster_score = cluster_scores.max()
        cluster_score = cluster_score if cluster_score > merge_minimum_score else merge_minimum_score
        merged_scores.append(cluster_score)

    merged_boxes = np.array(merged_boxes)
    merged_scores = np.array(merged_scores)
    merged_boxes_valid_idx = merged_boxes_valid_idx.cpu().numpy()

    return merged_boxes, merged_scores, merged_boxes_valid_idx


def merge_with_weighted_boxes_max_scores(boxes, scores, merge_iou='3D', merge_cnt=3, merge_iou_thr=0.5, merge_minimum_score=0.5):
    gpu_boxes = torch.from_numpy(boxes).to(torch.float32).cuda()
    gpu_scores = torch.from_numpy(scores).to(torch.float32).cuda()

    gpu_nms_boxes = nms_gpu(gpu_boxes, gpu_scores, thresh=0.1)
    nms_valid = torch.zeros(gpu_boxes.shape[0]).bool()
    nms_valid[gpu_nms_boxes[0]] = True

    if merge_iou == 'BEV':
        boxes_iou = boxes_iou_bev(gpu_boxes, gpu_boxes).cpu()
    else:
        boxes_iou = boxes_iou3d_gpu(gpu_boxes, gpu_boxes).cpu()

    boxes_iou_over_thr = (boxes_iou >= merge_iou_thr)
    boxes_iou_over_thr_cnt = boxes_iou_over_thr.sum(dim=0)
    boxes_iou_over_cnt_valid = boxes_iou_over_thr_cnt >= merge_cnt

    merge_boxes_valid = nms_valid * boxes_iou_over_cnt_valid
    merged_boxes_valid_idx = torch.where(merge_boxes_valid)[0]

    merged_boxes = []
    merged_scores = []
    ## no merged box, return
    if len(merged_boxes_valid_idx) < 1:
        return None, None, merged_boxes_valid_idx

    valid_clusters = {}
    cluster_cnt = 0
    for valid in merged_boxes_valid_idx:
        valid_cluster_idx = torch.where(boxes_iou_over_thr[valid])
        valid_cluster = {cluster_cnt: valid_cluster_idx}
        valid_clusters.update(valid_cluster)
        cluster_cnt += 1

    for ik in valid_clusters.keys():
        cluster_idx = valid_clusters[ik]
        cluster_boxes = boxes[cluster_idx]
        cluster_scores = scores[cluster_idx]
        cluster_sort_idx = cluster_scores.argsort()
        cluster_sort_idx = cluster_sort_idx[::-1]
        cluster_boxes = cluster_boxes[cluster_sort_idx]
        cluster_scores = cluster_scores[cluster_sort_idx]
        ## remove angle diff over thr
        max_scored_angle = cluster_boxes[0, 6]
        cluster_angle_diff = np.abs(cluster_boxes[:, 6] - max_scored_angle)
        valid_angle_idx = cluster_angle_diff < np.pi / 12.0
        cluster_boxes = cluster_boxes[valid_angle_idx]
        cluster_scores = cluster_scores[valid_angle_idx]
        ## computer weights
        weights = cluster_scores / cluster_scores.sum()
        # softmax
        # weights = torch.nn.function.softmax(weights)
        weighted_box = np.array([weights[i] * cluster_boxes[i] for i in range(len(weights))]).sum(axis=0)
        ## if use the max-scored direction
        #weighted_box[6] = cluster_boxes[0, 6]

        ## merge score cannot be weighted score
        #min_score = cluster_scores.min()
        #max_score = cluster_scores.max()
        #weighted_score = (weights*(max_score - min_score) + min_score).max()
        # merged score can be the max score
        merged_boxes.append(weighted_box)
        # setting the minimum score for merged boxes
        cluster_score = cluster_scores.max()
        cluster_score = cluster_score if cluster_score > merge_minimum_score else merge_minimum_score
        merged_scores.append(cluster_score)

    merged_boxes = np.array(merged_boxes)
    merged_scores = np.array(merged_scores)
    merged_boxes_valid_idx = merged_boxes_valid_idx.cpu().numpy()

    return merged_boxes, merged_scores, merged_boxes_valid_idx


def adjust_boxes_scores(boxes, scores, cfg_ps_adjust):
    ## decay the scores of the remote boxes
    ## PS_DECAY_FACTOR
    ## PS_VALID_DIST
    # score = score * (1-(dist/valid_dist)*decay_factor)
    ps_valid_dist = cfg_ps_adjust.get('PS_VALID_DIST', None)
    if ps_valid_dist is None:
        ps_valid_dist = 100.0
    centers = boxes[:, :3]
    centers_dist = np.square(centers).sum(axis=-1)
    valid_idx = centers_dist < np.square(ps_valid_dist)
    ## decay the scores

    dist_ratio = centers_dist/np.square(ps_valid_dist)
    ps_decay_factor = cfg_ps_adjust.get('PS_DECAY_FACTOR', None)
    if ps_decay_factor is None:
        ps_decay_factor = 0.0
    scores_factor = 1.0 - dist_ratio * ps_decay_factor
    decay_scores = scores_factor * scores
    decay_scores[valid_idx == False] = 0.0
    return decay_scores

def test_reproject_err():
    warp_errs = []
    for label in CUR_PSEUDO_LABELS:
        scan_info = CUR_PSEUDO_LABELS[label]
        ### verify the reprojection err
        warp_boxes_9 = scan_info["warp_boxes"].copy()
        det_boxes_9 = scan_info["gt_boxes"].copy()
        scan_pose = scan_info["pose"].cpu().numpy()
        scan_tr = scan_info["tr"].cpu().numpy()
        det_warp_back_boxes_9 = warp_global_to_loc(warp_boxes_9.copy(), scan_pose, scan_tr)
        warp_err = det_boxes_9 - det_warp_back_boxes_9
        warp_errs.append(warp_err)
    warp_errs_abs = np.fabs(np.concatenate(warp_errs, axis=0))
    return warp_errs_abs

def visualize_boxes_global(sequ_boxes, sequ_files=None, sequ_poses=None, pkl_file=None):
    # visualize det centers as points
    if pkl_file:
        # write in pkl, save for visualization
        filehandler = open(pkl_file, "wb")
        det_results = {
            "boxes": sequ_boxes,
            "files": sequ_files,
            "poses": sequ_poses
        }
        pickle.dump(det_results, filehandler)
        filehandler.close()
    return True

def visualize_points_global(points_global, pkl_file=None, ply_file=None):

    if ply_file:
        point_color = np.repeat(np.reshape([255, 255, 255], [1, -1]), points_global.shape[0], axis=0)
        points_ply = np.concatenate([points_global[:, :3], point_color], axis=-1)
        write_ply(points_ply, ply_file)

    if pkl_file:
        # write in pkl, save for visualization
        filehandler = open(pkl_file, "wb")
        det_results = {
            "points_global": points_global,
        }
        pickle.dump(det_results, filehandler)
        filehandler.close()
    return True

#def evaluate_poses(sequences, sequences_poses):
#
#    for sequ in sequences:
#        # tr: transform from cam0 to lidar
#        tr = np.identity(4)
#        tr[:3, :] = sequences_poses[sequ]["calib"]["Tr"]
#        # poses
#        poses = [pose for pose in sequences_poses[sequ]["poses"]]
#        poses = np.array(poses)
#        assert len(poses.shape) == 3
#        pose_homo = np.identity(4)
#        pose_homo = pose_homo.reshape(-1, 4, 4)
#        poses_homo = np.repeat(pose_homo/home/zc/3d-detection/ST3D_LOCAL/tools/merge_result/tmp_kitti_result, poses.shape[0], axis=0)
#        poses_homo[:, :3, :] = poses
#
#        lidar_poses_homo_list = [np.matmul(poses_homo[i], tr) for i in range(poses_homo.shape[0])]
#        lidar_poses_homo = np.array(lidar_poses_homo_list)
#        # return to lidar coordinate system
#        lidar_coor_poses_homo_list = [np.matmul(np.linalg.inv(tr), lidar_poses_homo[i]) for i in
#                                      range(lidar_poses_homo.shape[0])]
#        lidar_coor_poses_homo = np.array(lidar_coor_poses_homo_list)
#        lidar_poses = lidar_coor_poses_homo[:, :, 3]
#
#        ## test eular angle compute order
#    return True
def global_merge_and_verify_boxes(rank=0, glbmem_update=False, cur_epoch=None):

    if rank==0:
        print("merge and verify ps boxes via transform consistency:")
    ## parsing sequences
    sequences = []
    for label in RAW_PSEUDO_LABELS:
        sequ = MERGE_PARS[label]['sequ_id']
        if sequ not in sequences:
            sequences.append(sequ)
    if rank == 0:
        print("{} sequences found, merging...".format(len(sequences)))

    ## init sequence-wise boxes
    sequ_det_boxes = {}
    sequ_det_scores = {}
    sequ_merged_boxes = {}
    sequ_warp_errs = {}
    sequ_lidar_files = {}
    sequ_poses_tr = {}
    for sequ in sequences:
        sequ_det_boxes.update({sequ: []})
        sequ_det_scores.update({sequ: []})
        sequ_merged_boxes.update({sequ: []})
        sequ_warp_errs.update({sequ: []})
        sequ_lidar_files.update({sequ: []})
        sequ_poses_tr.update({sequ: []})

    ## reorganize and warp the detection along sequences in global coors
    for label in RAW_PSEUDO_LABELS:
        scan_info = RAW_PSEUDO_LABELS[label]
        det_boxes_9 = scan_info["gt_boxes"].copy()
        merge_info = MERGE_PARS[label].copy()
        scan_pose = merge_info["pose"]
        scan_tr = merge_info["tr"]

        if cfg.get('DATA_CONFIG_TAR.SHIFT_COOR', None):
            det_boxes_9[:, 0:3] -= np.array(cfg.DATA_CONFIG_TAR.SHIFT_COOR, dtype=np.float32)

        if cfg.SELF_TRAIN.get('MERGE_SCORE', None) == "cls":
            scores = scan_info["cls_scores"]
        else:
            scores = scan_info["iou_scores"]

        ## adjust detected boxes according to distance, points, etc, oFalsefor merge
        if cfg.SELF_TRAIN.get('PS_ADJUST', None) and cfg.SELF_TRAIN.PS_ADJUST.ENABLE :
            adjusted_scores = adjust_boxes_scores(det_boxes_9, scores.copy(), cfg.SELF_TRAIN.get('PS_ADJUST', None))
        else:
            adjusted_scores = scores

        # remove boxes beyond the valid distance
        valid_idx = adjusted_scores > 0.0001
        adjusted_scores = adjusted_scores[valid_idx]
        det_boxes_9 = det_boxes_9[valid_idx]

        ## warp to global
        warp_boxes_9 = warp_loc_to_global(det_boxes_9, scan_pose, scan_tr)

        sequ = merge_info['sequ_id']
        sequ_det_boxes[sequ].append(warp_boxes_9)
        sequ_det_scores[sequ].append(adjusted_scores)
        sequ_lidar_files[sequ].append(label)
        sequ_poses_tr[sequ].append({'pose': scan_pose, 'tr': scan_tr})

    ## some verification tests for debug
    #reproject_err = test_reproject_err()
    #print("reproject err: {}".format(reproject_err))

    ## merge the boxes in global coords by nms to generate global static boxes
    merge_iou_thr = cfg.SELF_TRAIN.get('MERGE_IOU_THR', None)
    merge_cnt = cfg.SELF_TRAIN.get('MERGE_CNT', None)

    pos_glb_ps_meter = common_utils.AverageMeter()
    ign_glb_ps_meter = common_utils.AverageMeter()

    for sequ in sequences:
        sequ_det_boxes[sequ] = np.concatenate(sequ_det_boxes[sequ], axis=0)
        det_boxes = sequ_det_boxes[sequ]
        det_scores = np.concatenate(sequ_det_scores[sequ], axis=0)

        if cfg.SELF_TRAIN.ONLY_POS_PS:
            valid_index = det_boxes[:, 7] > 0
            boxes_valid = det_boxes[valid_index]
            scores_valid = det_scores[valid_index]
        else:
            boxes_valid = det_boxes
            scores_valid = det_scores

        if boxes_valid.shape[0] > 0:
            # merge boxes in global frames for each sequence
            if cfg.SELF_TRAIN.get('MERGE_TYPE', None) == 'MAX':
                merged_boxes, merged_scores, merged_boxes_index = merge_with_max_boxes_max_scores(
                    boxes_valid[:, :7],
                    scores_valid,
                    merge_iou=cfg.SELF_TRAIN.get('MERGE_IOU',None),
                    merge_iou_thr=merge_iou_thr,
                    merge_cnt=merge_cnt,
                    merge_minimum_score=cfg.SELF_TRAIN.get('MERGE_MINIMUM_SCORE', None)
                )
            else:
                merged_boxes, merged_scores, merged_boxes_index = merge_with_weighted_boxes_max_scores(
                    boxes_valid[:, :7],
                    scores_valid,
                    merge_iou=cfg.SELF_TRAIN.get('MERGE_IOU', None),
                    merge_iou_thr=merge_iou_thr,
                    merge_cnt=merge_cnt,
                    merge_minimum_score=cfg.SELF_TRAIN.get('MERGE_MINIMUM_SCORE', None)
                )

            if merged_boxes is not None and merged_boxes.shape[0]>0:
                # update boxes valid
                boxes_valid[merged_boxes_index, :7] = merged_boxes
                boxes_valid[merged_boxes_index, 8] = merged_scores
                boxes_valid[merged_boxes_index, 7] = 1
                scores_valid[merged_boxes_index] = merged_scores
            merged_boxes = boxes_valid[merged_boxes_index]
            merged_scores = scores_valid[merged_boxes_index]
        else:
            merged_boxes = boxes_valid
            print("num of boxes_valid = {}".format(boxes_valid.shape[0]))
            merged_scores = scores_valid

        # ensemble global pseudo label to global memory
        gm_infos = {
            'gt_boxes': merged_boxes,
            'cls_scores': None,
            'iou_scores': merged_scores,
            'memory_counter': np.zeros(merged_boxes.shape[0])
        }

        if cur_epoch == 0:
            GLBMEM_LABELS[sequ]=gm_infos
        else:
            if glbmem_update:
                # update GLBMEM_LABELS at each epoch
                assert len(sequences) == len(GLBMEM_LABELS)
                ensemble_func = getattr(memory_ensemble_gm_utils, cfg.SELF_TRAIN.GLBMEM_ENSEMBLE.NAME)
                gm_emsamble_infos = ensemble_func(GLBMEM_LABELS[sequ], gm_infos, cfg.SELF_TRAIN.GLBMEM_ENSEMBLE)
                # update the labels of pos and ign
                scores = gm_emsamble_infos['gt_boxes'][:, 8]
                valid_idx = scores > cfg.SELF_TRAIN.IOU_SCORE_THRESH
                invalid_idx = scores < cfg.SELF_TRAIN.IOU_SCORE_THRESH
                gm_emsamble_infos['gt_boxes'][valid_idx, 7] = 1
                gm_emsamble_infos['gt_boxes'][invalid_idx, 7] = -1
                GLBMEM_LABELS[sequ] = gm_emsamble_infos
            else:
                GLBMEM_LABELS[sequ] = gm_infos

        gm_infos = GLBMEM_LABELS[sequ]
        assert (gm_infos['gt_boxes'][:, 7] < 0).sum() == (gm_infos['gt_boxes'][:, 8] < cfg.SELF_TRAIN.IOU_SCORE_THRESH).sum()
        pos_glb_ps_meter.update((gm_infos['gt_boxes'][:, 7] > 0).sum())
        ign_glb_ps_meter.update(gm_infos['gt_boxes'].shape[0] - (gm_infos['gt_boxes'][:, 7] > 0).sum())
        disp_dict = {'pos_global_ps_box': "{:.3f} / per sequ".format(pos_glb_ps_meter.avg),
                     'ign_global_ps_box': "{:.3f} / per sequ".format(ign_glb_ps_meter.avg)}

    if rank == 0:
        print("global merging done!")
        print(disp_dict)

    return pos_glb_ps_meter.avg, ign_glb_ps_meter.avg

def merge_two_pathway_boxes(rank=0, cur_epoch=None):

    ## 1) verify the detected boxes in global static boxes and 2) warp the verifyed boxes back to local coors
    static_ps_meter = common_utils.AverageMeter()
    moving_ps_meter = common_utils.AverageMeter()

    if cfg.SELF_TRAIN.get('LOCMEM_ENSEMBLE', None) and cur_epoch > 0:
        # add local memory
        CUR_PSEUDO_LABELS.update(LOCMEM_LABELS)
    else:
        # disable local memory
        CUR_PSEUDO_LABELS.update(RAW_PSEUDO_LABELS)

    for label in CUR_PSEUDO_LABELS:
        scan_info = CUR_PSEUDO_LABELS[label]
        det_boxes_9 = scan_info["gt_boxes"]
        # SHIFT before warp to global
        if cfg.get('DATA_CONFIG_TAR.SHIFT_COOR', None):
            det_boxes_9[:, 0:3] -= np.array(cfg.DATA_CONFIG_TAR.SHIFT_COOR, dtype=np.float32)

        merge_info = MERGE_PARS[label]
        scan_pose = merge_info["pose"]
        scan_tr = merge_info["tr"]
        warp_boxes_9 = warp_loc_to_global(det_boxes_9, scan_pose, scan_tr)

        sequ = merge_info['sequ_id']
        merged_boxes = GLBMEM_LABELS[sequ]['gt_boxes']
        # verify labels
        if merged_boxes.shape[0]>0 and warp_boxes_9.shape[0]>0:
            idx_verify_warp, idx_verify_merged = verify_labels_iou3d(warp_boxes_9[:, :7],
                                                                     merged_boxes[:, :7],
                                                                     merge_iou=cfg.SELF_TRAIN.get('MERGE_IOU'),
                                                                     verify_thr=cfg.SELF_TRAIN.get('VERIFY_IOU_THR')
                                                                     )
            warp_verify_boxes_9 = warp_boxes_9.copy()
            if idx_verify_merged.shape[0] > 0:
                static_ps_meter.update(idx_verify_merged.shape[0])
                moving_ps_meter.update(warp_boxes_9.shape[0] - idx_verify_merged.shape[0])
                change_merged_boxes = merged_boxes[idx_verify_merged]
                # set a sign of verified boxes
                change_merged_boxes[:, 7] = 2
                warp_verify_boxes_9[idx_verify_warp] = change_merged_boxes
                ## restore the ignored label(-1)
                warp_verify_boxes_9[warp_boxes_9[:, 7] == -1, 7] = -1
            ## warp back to local coors and update ps labels
            det_verify_boxes_9 = warp_global_to_loc(warp_verify_boxes_9.copy(), scan_pose, scan_tr)
            # SHIFT after warp to local
            if cfg.get('DATA_CONFIG_TAR.SHIFT_COOR', None):
                det_verify_boxes_9[:, 0:3] += np.array(cfg.DATA_CONFIG_TAR.SHIFT_COOR, dtype=np.float32)
            CUR_PSEUDO_LABELS[label]["gt_boxes"] = det_verify_boxes_9

            ## computer verify error
            local_det_diff = np.fabs(det_verify_boxes_9 - det_boxes_9)
            local_det_err = local_det_diff[:, :6].sum(axis=0) / float(local_det_diff.shape[0])
            CUR_PSEUDO_LABELS[label]['det_ver_err'] = local_det_err

        else:
            CUR_PSEUDO_LABELS[label]['det_ver_err'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if rank == 0:
                print("Global merge fail: scan {} has {} boxes, sequ {} has {} merged boxes.".format(label, warp_boxes_9.shape[0], sequ, merged_boxes.shape[0]))

    if rank == 0:
        print("merge and verify done! avg of static boxes: {}, avg of moving boxes {}. ".format(static_ps_meter.avg, moving_ps_meter.avg))

    return static_ps_meter.avg, moving_ps_meter.avg


def global_pathway_pseudo_label(rank, cur_epoch):

    # add transformation merge operation
    glb_ps_pos, glb_ps_ign = global_merge_and_verify_boxes(rank,
                                  glbmem_update=(cfg.SELF_TRAIN.get('GLBMEM_ENSEMBLE', None) and
                                                 cfg.SELF_TRAIN.GLBMEM_ENSEMBLE.ENABLED and
                                                 cur_epoch > 0),
                                   cur_epoch=cur_epoch
                                  )
    loc_ps_static, loc_ps_moving = merge_two_pathway_boxes(rank, cur_epoch=cur_epoch)

    return {
        'avg_glb_ps_pos': glb_ps_pos,
        'avg_glb_ps_ign': glb_ps_ign,
        'avg_loc_ps_static': loc_ps_static,
        'avg_loc_ps_moving': loc_ps_moving,
    }


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()
    # dump new pseudo label to given dir

    PSEUDO_LABELS.update(CUR_PSEUDO_LABELS)
    CUR_PSEUDO_LABELS.clear()
    MERGE_PARS.clear()

    if not cfg.SELF_TRAIN.LOCMEM_ENSEMBLE.ENABLED:
        LOCMEM_LABELS.clear()
    else:
        if cur_epoch == 0 and LOCMEM_LABELS.__len__() == 0 :
            LOCMEM_LABELS.update(PSEUDO_LABELS)

    if not cfg.SELF_TRAIN.GLBMEM_ENSEMBLE.ENABLED:
        GLBMEM_LABELS.clear()

    print("pseudo labels total %d iterms" % len(PSEUDO_LABELS))

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(PSEUDO_LABELS, f)

        if cfg.SELF_TRAIN.GLBMEM_ENSEMBLE.ENABLED:
            ps_path = os.path.join(ps_label_dir, "ps_glb_label_e{}.pkl".format(cur_epoch))
            with open(ps_path, 'wb') as f:
                pkl.dump(GLBMEM_LABELS, f)

def local_pathway_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            locmem_update=True):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        locmem_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    loc_ps_pos_meter = common_utils.AverageMeter()
    loc_ps_ign_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()

            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            local_merge_score = cfg.SELF_TRAIN.get('LOC_MERGE_SCORE', None)
            if local_merge_score == 'cls':
                pred_scores = pred_cls_scores
                ## check pred scores
                # scores_err = np.abs(pred_scores - pred_scores_from_model).sum()
                # assert scores_err < 0.1
                # remove boxes under negative threshold
                if cfg.SELF_TRAIN.get('CLS_NEG_THRESH', None):
                    labels_remove_scores = np.array(cfg.SELF_TRAIN.CLS_NEG_THRESH)[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    pred_boxes = pred_boxes[remain_mask]
                    if 'pred_cls_scores' in pred_dicts[b_idx]:
                        pred_cls_scores = pred_cls_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]
                labels_ignore_scores = np.array(cfg.SELF_TRAIN.CLS_SCORE_THRESH)[pred_labels - 1]
                ignore_mask = pred_scores < labels_ignore_scores
                pred_labels[ignore_mask] = -1

            elif local_merge_score == 'iou':
                pred_scores = pred_iou_scores
                ## check pred scores
                # scores_err = np.abs(pred_scores - pred_scores_from_model).sum()
                # assert scores_err < 0.1
                # remove boxes under negative threshold
                if cfg.SELF_TRAIN.get('IOU_NEG_THRESH', None):
                    labels_remove_scores = np.array(cfg.SELF_TRAIN.IOU_NEG_THRESH)[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    pred_boxes = pred_boxes[remain_mask]
                    if 'pred_cls_scores' in pred_dicts[b_idx]:
                        pred_cls_scores = pred_cls_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]
                labels_ignore_scores = np.array(cfg.SELF_TRAIN.IOU_SCORE_THRESH)[pred_labels - 1]
                ignore_mask = pred_scores < labels_ignore_scores
                pred_labels[ignore_mask] = -1

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }

        gt_merge_infos = {}
        if cfg.DATA_CONFIG_TAR.DATASET == 'NuScenesOdomDataset' or cfg.DATA_CONFIG_TAR.DATASET == 'KittiOdomDataset':
            gt_merge_infos.update({'sequ_id': input_dict['sequ_id'][b_idx]})
            gt_merge_infos.update({'tr': input_dict['tr'][b_idx]})
            gt_merge_infos.update({'pose': input_dict['pose'][b_idx]})
        else:
            raise ValueError('Dataset should be OdomDataset, %s not support.' % (cfg.DATA_CONFIG_TAR.DATASET))

        # store pesudo labels
        RAW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos
        # store pars for trans merge
        MERGE_PARS[input_dict['frame_id'][b_idx]] = gt_merge_infos




        # record pseudo label to pseudo label memory
        if locmem_update:
            ensemble_func = getattr(memory_ensemble_gm_utils, cfg.SELF_TRAIN.LOCMEM_ENSEMBLE.NAME)
            loc_ensamble_infos = ensemble_func(LOCMEM_LABELS[input_dict['frame_id'][b_idx]],
                                     gt_infos, cfg.SELF_TRAIN.LOCMEM_ENSEMBLE)
            LOCMEM_LABELS[input_dict['frame_id'][b_idx]] = loc_ensamble_infos

            # update the meters
            loc_ps_ign_meter.update((loc_ensamble_infos['gt_boxes'][:, 7] == -1).sum())
            loc_ps_pos_meter.update((loc_ensamble_infos['gt_boxes'][:, 7] == 1).sum())
        else:
            loc_ps_ign_meter.update((gt_infos['gt_boxes'][:, 7] == -1).sum())
            loc_ps_pos_meter.update((gt_infos['gt_boxes'][:, 7] == 1).sum())

    return loc_ps_pos_meter.avg, loc_ps_ign_meter.avg

def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """

    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
        if 'det_ver_err' in PSEUDO_LABELS[frame_id]:
            det_ver_err = PSEUDO_LABELS[frame_id]['det_ver_err']
        else:
            det_ver_err = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    else:
        print('length of PSEUDO_LABELS is {}'.format(len(PSEUDO_LABELS.keys())))
        raise ValueError('Cannot find pseudo label for frame: %s in total %d items' % (frame_id, len(PSEUDO_LABELS)))

    return gt_box, det_ver_err
