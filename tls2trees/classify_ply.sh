#!/bin/bash

# required for bunya (adjust as necessary)
module load cuda/11.8.0
# source activate tls2trees

MODEL_PATH="/home/uqtdeve1/veg3d/uqtdeve1/code/TLS2trees/tls2trees/fsct/model/model_base.pth"
BATCH_SIZE=16
NUM_PROCS=16

# Base directory containing all .ply files
BASE_DIR="/home/uqtdeve1/Q5866/uqtdeve1/Nextcloud/data/chapter_3/epi_trees/points"

# Process each .ply file in BASE_DIR
for point_cloud in "$BASE_DIR"/*.ply; do

    # Make sure the file actually exists (in case the glob is empty)
    if [ -f "$point_cloud" ]; then
        # Extract filename (without path and extension)
        filename=$(basename "$point_cloud" .ply)

        echo "Processing: $point_cloud"

        echo "Running base model..."
        CMD="python3 semantic.py \
            --point-cloud \"$point_cloud\" \
            --odir \"$BASE_DIR/$filename\" \
            --params \"$BASE_DIR/${filename}/${filename}.params.pickle\" \
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
