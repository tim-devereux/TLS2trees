#!/bin/bash

# required for bunya
# module load cuda/11.8.0
# source activate tls2trees

# Default values
INPUT_DIR="/home/tim/Code/TLS2trees/test_data"

MODEL_PATH="/home/tim/Code/TLS2trees/tls2trees/fsct/model/model_base.pth"
BATCH_SIZE=4
NUM_PROCS=8

# Process each .ply file in the input directory
for point_cloud in "$INPUT_DIR"/*.ply; do
    if [ -f "$point_cloud" ]; then
        # Extract filename without path and extension
        filename=$(basename "$point_cloud" .ply)        
        echo "Processing: $filename"
        
        # Process
        echo "Running base model..."
        CMD="python3 semantic.py \
            --point-cloud $point_cloud \
            --batch_size $BATCH_SIZE \
            --num_procs $NUM_PROCS \
            --model $MODEL_PATH \
            --keep-npy \
            --verbose"
        eval $CMD
        
        echo "Completed processing: $filename"
        echo "----------------------------------------"
    fi
done

echo "All point clouds processed successfully!"