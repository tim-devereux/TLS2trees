#!/bin/bash

# required for bunya
module load cuda/11.8.0
source activate tls2trees

# Default values
# INPUT_DIR="/home/uqtdeve1/code/TLS2trees/test_data"
INPUT_DIR="/home/uqtdeve1/veg3d/TERN_TLS_DATA/final/GWW_2021"

# MODEL_PATH="/home/tim/Code/TLS2trees/tls2trees/fsct/model/model_base.pth"
MODEL_PATH="/home/uqtdeve1/code/TLS2trees/tls2trees/fsct/model/model_base.pth"
BATCH_SIZE=16
NUM_PROCS=32

# Process each .ply file in the input directory
for point_cloud in "$INPUT_DIR"/*_raycloud_bound.ply; do
    if [ -f "$point_cloud" ]; then
        # Extract filename without path and extension
        filename=$(basename "$point_cloud" .ply)        
        echo "Processing: $point_cloud"

        mkdir -p $INPUT_DIR/${filename}_npy
        
        # # Process all 
        # echo "Running base model..."
        # CMD="python3 semantic.py \
        #     --point-cloud $point_cloud \
        #     --odir $INPUT_DIR/$filename \
        #     --batch_size $BATCH_SIZE \
        #     --num_procs $NUM_PROCS \
        #     --model $MODEL_PATH \
        #     --keep-npy \
        #     --step all \
        #     --verbose"
        # eval $CMD

        # # Process
        echo "Running base model..."
        CMD="python3 semantic.py \
            --point-cloud $point_cloud \
            --odir $INPUT_DIR/$filename \
            --params $INPUT_DIR/{$filename}_npy/$filename.params.pickle \
            --batch_size $BATCH_SIZE \
            --num_procs $NUM_PROCS \
            --model $MODEL_PATH \
            --keep-npy \
            --step all \
            --verbose"
        eval $CMD
        
        echo "Completed processing: $filename"
        echo "----------------------------------------"
    fi
done

echo "All point clouds processed successfully!"