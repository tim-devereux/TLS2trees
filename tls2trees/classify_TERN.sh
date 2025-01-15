#!/bin/bash

# required for bunya
module load cuda/11.8.0
# source activate tls2trees

MODEL_PATH="/home/uqtdeve1/veg3d/uqtdeve1/code/TLS2trees/tls2trees/fsct/model/model_base.pth"
BATCH_SIZE=16
NUM_PROCS=16

# Base directory
BASE_DIR="/home/uqtdeve1/veg3d/TERN_TLS_DATA/final"

# List of sites (comment out any sites you wish to skip)
sites=(
  # "LITCH_2021"
  # "GINGIN_2021"
  # "ROBSON_2023"
  # "ALICE_2021"
  "TUMBA_2022"
#   "CALP_2021"
  # "CUP_2022"
  # "GWW_2021"
)



# Loop through each site in the array
for site in "${sites[@]}"; do

    # Construct the input directory for this site
    INPUT_DIR="$BASE_DIR/$site"

    # Process each .ply file in the input directory
    for point_cloud in "$INPUT_DIR"/*_raycloud_bound.ply; do

        if [ -f "$point_cloud" ]; then
            # Extract filename without path and extension
            filename=$(basename "$point_cloud" .ply)
            echo "Processing: $point_cloud"

            echo "Running base model..."
            CMD="python3 semantic.py \
                --point-cloud \"$point_cloud\" \
                --odir \"$INPUT_DIR/$filename\" \
                --params \"$INPUT_DIR/${filename}/${filename}.params.pickle\" \
                --batch_size $BATCH_SIZE \
                --num_procs $NUM_PROCS \
                --model $MODEL_PATH \
                --keep-npy \
                --step all \
                --verbose"
            eval "$CMD"

            echo "Completed processing: $filename"
            echo "----------------------------------------"
        fi

    done
done
