import _init_path
import os
import torch
from tensorboardX import SummaryWriter
import time
import glob
import re
import datetime
import argparse
from pathlib import Path
import torch.distributed as dist
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from eval_utils import eval_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, required=False, help='Number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--ps_dir', type=str, default=None, help='specify a ps directory to be evaluated if needed')
    parser.add_argument('--ps_epoch', type=int, default=0, help='ps_epoch')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    import pickle as pkl
    args, cfg = parse_config()

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_ps_output_dir = output_dir / 'eval_ps'
    eval_ps_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_ps_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    #for key, val in vars(args).items():
    #    logger.info('{:16} {}'.format(key, val))
    #log_config_to_file(cfg, logger=logger)

    if cfg.get('DATA_CONFIG_TAR', None):
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_TAR,
            class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=None, workers=args.workers, logger=logger, training=True
        )

    dataset = test_loader.dataset
    class_names = dataset.class_names

    ps_label_dir = args.ps_dir
    cur_epoch = args.ps_epoch
    ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
    with open(ps_path, 'rb') as f:
        PSEUDO_LABELS = pkl.load(f)

    det_annos = PSEUDO_LABELS
    result_str, result_dict = dataset.evaluation_ps(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=eval_ps_output_dir
    )
    print(result_str)

if __name__ == '__main__':
    main()

