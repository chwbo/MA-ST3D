#!/bin/sh

export CUDA_VISIBLE_DEVICES=1,2,3,4

bash scripts/dist_test_odom.sh 4 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d_odom/pvrcnn_st3d_odom.yaml --batch_size 8 --eval_all --extra_tag sequ_50_pvrcnn_0

bash scripts/dist_test.sh 4 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml --batch_size 8 --eval_all --extra_tag base_pvrcnn_st_0
