#!/bin/bash
module load cuda/11.8.0
source activate tls2trees


TEST_DATA="plot_tiles"

# Default values
INPUT_DIR="/scratch/project/veg3d/uqtdeve1/training_data/test_data/$TEST_DATA"

# SYNTH_MODEL_PATH="/scratch/project/veg3d/uqtdeve1/training_data/models/model_checkpoints/best_model.pth"
SYNTH_MODEL_PATH="/scratch/project/veg3d/uqtdeve1/training_data/models/model_checkpoints/checkpoint_epoch_1940.pth"
SYNTH_OUTPUT_PATH="/scratch/project/veg3d/uqtdeve1/training_data/test_data/$TEST_DATA/synth_results"
BASE_OUTPUT_PATH="/scratch/project/veg3d/uqtdeve1/training_data/test_data/$TEST_DATA/base_results"
BASE_MODEL_PATH="/scratch/project/veg3d/uqtdeve1/training_data/models/model_base.pth"
BATCH_SIZE=4
NUM_PROCS=32

# Create output directories if they don't exist
mkdir -p "$SYNTH_OUTPUT_PATH"
mkdir -p "$BASE_OUTPUT_PATH"

# Process each .ply file in the input directory
for point_cloud in "$INPUT_DIR"/*.ply; do
    if [ -f "$point_cloud" ]; then
        # Extract filename without path and extension
        filename=$(basename "$point_cloud" .ply)
        
        echo "Processing: $filename"
        
        # Create output subdirectories for this file
        synth_subdir="$SYNTH_OUTPUT_PATH/$filename"
        base_subdir="$BASE_OUTPUT_PATH/$filename"
        mkdir -p "$synth_subdir"
        mkdir -p "$base_subdir"
        
        # Process with synthetic model
        echo "Running synthetic model..."
        CMD="python3 semantic.py \
            --point-cloud '$point_cloud' \
            --batch_size $BATCH_SIZE \
            --num_procs $NUM_PROCS \
            --model $SYNTH_MODEL_PATH \
            --odir $synth_subdir \
            --verbose"
        eval $CMD
        
        # # Process with base model
        # echo "Running base model..."
        # CMD="python3 semantic.py \
        #     --point-cloud '$point_cloud' \
        #     --batch_size $BATCH_SIZE \
        #     --num_procs $NUM_PROCS \
        #     --model $BASE_MODEL_PATH \
        #     --odir $base_subdir \
        #     --verbose"
        # eval $CMD
        
        # echo "Completed processing: $filename"
        # echo "----------------------------------------"
    fi
done

echo "All point clouds processed successfully!"