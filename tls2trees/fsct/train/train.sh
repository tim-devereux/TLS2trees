#!/bin/bash

# Load CUDA module
module load cuda/11.8.0

# Define the directory containing .ply files
TRAIN_DATA_PATH="/scratch/project/veg3d/uqtdeve1/training_data/single_trees_synthetic/train/detailed"
MODEL_PATH="/scratch/project/veg3d/uqtdeve1/training_data/models/model_base_synth.pth"
SAVE_CHUNKS_DIR="/scratch/project/veg3d/uqtdeve1/training_data/models/"

# Get all .ply files in the training directory
files=("$TRAIN_DATA_PATH"/*.ply)

# Get the total number of files
num_files=${#files[@]}

# Iterate over all .ply files in the directory
for train_file in "${files[@]}"; do
    # Check if file exists
    if [ -f "$train_file" ]; then
        echo "Processing $train_file..."
        
        # Randomly select a different file for validation
        while true; do
            # Generate random index
            random_idx=$((RANDOM % num_files))
            validate_file="${files[$random_idx]}"
            
            # Make sure validation file is different from training file
            if [ "$validate_file" != "$train_file" ]; then
                break
            fi
        done
        
        echo "Using $validate_file for validation..."
        
        python train.py \
            --train "$train_file" \
            --model "$MODEL_PATH" \
            --validate "$validate_file" \
            --label classification \
            --iterations 1000 \
            --save-chunks-to "$SAVE_CHUNKS_DIR" \
            --verbose
    fi
done