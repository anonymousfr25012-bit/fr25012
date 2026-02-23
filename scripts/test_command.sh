#!/bin/bash

# Common parameters
DATA_DIR_1="../data/"


MODEL="../models/test_model.pt"


COMMON_ARGS="--fisheye --feature FPFH --cam_z -0 --cam_x 0 --cam_y 0 --coffset_x 0 --coffset_y 0 --coffset_z 0 --coffset_yaw 0 --cam_x_gt 0 --cam_y_gt 0 --cam_z_gt 0 --ransac_n_closest --ransac_forward_reverse --pt_type XYZRGB --icp_refine --masking_distance -1"

# Function to run matcher
run_matcher_1() {
    local scan1=$1
    local scan2=$2
    ./matcher --input "${DATA_DIR_1}/FARO_Scan_${scan1}_1cm.pcd" \
              --input_2 "${DATA_DIR_1}/FARO_Scan_${scan2}_1cm.pcd" \
              --pos_file_1 "${DATA_DIR_1}/FARO_Scan_${scan1}_1cm_global_pose.txt" \
              --pos_file_2 "${DATA_DIR_1}/FARO_Scan_${scan2}_1cm_global_pose.txt" \
              --model "${MODEL}"  \
              ${COMMON_ARGS}
}



# Execute commands
run_matcher_1 035 034
run_matcher_1 055 054
