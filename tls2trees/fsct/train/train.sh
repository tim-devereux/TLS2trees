#!/bin/bash

# Configuration
CUDA_MODULE="cuda/11.8.0"
TRAIN_DATA_PATH="/scratch/project/veg3d/uqtdeve1/training_data/single_trees_synthetic/train/detailed"
MODEL_PATH="/scratch/project/veg3d/uqtdeve1/training_data/models/model_base_synth.pth"
SAVE_CHUNKS_DIR="/scratch/project/veg3d/uqtdeve1/training_data/models/"
NUM_FILES_TO_PROCESS=10  # Adjust this number as needed
ITERATIONS=1000

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if directories exist
check_directories() {
    local dirs=("$TRAIN_DATA_PATH" "$(dirname "$MODEL_PATH")" "$SAVE_CHUNKS_DIR")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_message "ERROR: Directory not found: $dir"
            exit 1
        fi
    done
}

# Function to get random file
get_random_file() {
    local exclude_file="$1"
    local files=("$TRAIN_DATA_PATH"/*.ply)
    local num_files=${#files[@]}
    local max_attempts=100
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        random_idx=$((RANDOM % num_files))
        selected_file="${files[$random_idx]}"
        
        if [ "$selected_file" != "$exclude_file" ]; then
            echo "$selected_file"
            return 0
        fi
        ((attempt++))
    done
    
    log_message "ERROR: Could not find a different random file after $max_attempts attempts"
    exit 1
}

# Main execution
main() {
    # Load CUDA module
    log_message "Loading CUDA module $CUDA_MODULE"
    module load "$CUDA_MODULE"

    # Check directories
    check_directories

    # Get list of all PLY files
    files=("$TRAIN_DATA_PATH"/*.ply)
    total_files=${#files[@]}

    if [ $total_files -eq 0 ]; then
        log_message "ERROR: No .ply files found in $TRAIN_DATA_PATH"
        exit 1
    fi

    log_message "Found $total_files PLY files in training directory"
    log_message "Will process $NUM_FILES_TO_PROCESS random files"

    # Create array of random indices
    indices=($(seq 0 $((total_files-1)) | shuf | head -n $NUM_FILES_TO_PROCESS))

    # Process random files
    for idx in "${indices[@]}"; do
        train_file="${files[$idx]}"
        
        log_message "Processing training file: $(basename "$train_file")"
        
        # Get random validation file
        validate_file=$(get_random_file "$train_file")
        log_message "Selected validation file: $(basename "$validate_file")"
        
        # Run training
        log_message "Starting training iteration..."
        python train.py \
            --train "$train_file" \
            --model "$MODEL_PATH" \
            --validate "$validate_file" \
            --label classification \
            --iterations $ITERATIONS \
            --save-chunks-to "$SAVE_CHUNKS_DIR" \
            --verbose
            
        if [ $? -eq 0 ]; then
            log_message "Successfully completed training with $(basename "$train_file")"
        else
            log_message "ERROR: Training failed for $(basename "$train_file")"
        fi
        
        echo "----------------------------------------"
    done

    log_message "All training iterations completed"
}

# Execute main function with error handling
{
    main
} 2>&1 | tee -a "${SAVE_CHUNKS_DIR}/training_log_$(date '+%Y%m%d_%H%M%S').log"