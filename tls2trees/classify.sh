#!/bin/bash

# Default values
POINT_CLOUD="/mnt/NVME/plot_recon/LITCH_2022/LITCH_2022_raycloud_bound_decimated.ply"
OUTPUT_DIR="./output"
BATCH_SIZE=4
NUM_PROCS=8
STEP=3
IS_WOOD=1
OUTPUT_FORMAT="ply"

# Function to display usage
usage() {
    echo "Usage: $0 -p <point_cloud_path> [options]"
    echo
    echo "Required arguments:"
    echo "  -p, --point-cloud     Path to point cloud file"
    echo
    echo "Optional arguments:"
    echo "  -o, --output-dir      Output directory (default: ./output)"
    echo "  -b, --batch-size      Batch size for processing (default: 10)"
    echo "  -n, --num-procs       Number of CPU cores to use (default: 10)"
    echo "  -s, --step            Which process to run to (default: 3)"
    echo "  -w, --is-wood         Probability threshold for wood classification (default: 1.0)"
    echo "  -f, --format          Output format (default: ply)"
    echo "  -t, --tile-index      Path to tile index file"
    echo "  --buffer              Buffer size for neighbouring tiles"
    echo "  --keep-npy           Keep .npy files after processing"
    echo "  --model              Path to custom model"
    echo "  --verbose            Enable verbose output"
    echo "  -h, --help           Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -p | --point-cloud)
        POINT_CLOUD="$2"
        shift 2
        ;;
    -o | --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
    -b | --batch-size)
        BATCH_SIZE="$2"
        shift 2
        ;;
    -n | --num-procs)
        NUM_PROCS="$2"
        shift 2
        ;;
    -s | --step)
        STEP="$2"
        shift 2
        ;;
    -w | --is-wood)
        IS_WOOD="$2"
        shift 2
        ;;
    -f | --format)
        OUTPUT_FORMAT="$2"
        shift 2
        ;;
    -t | --tile-index)
        TILE_INDEX="$2"
        shift 2
        ;;
    --buffer)
        BUFFER="$2"
        shift 2
        ;;
    --keep-npy)
        KEEP_NPY="--keep-npy"
        shift
        ;;
    --model)
        MODEL="$2"
        shift 2
        ;;
    --verbose)
        VERBOSE="--verbose"
        shift
        ;;
    -h | --help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Check if point cloud path is provided
if [ -z "$POINT_CLOUD" ]; then
    echo "Error: Point cloud path is required"
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Construct the command
CMD="python3 semantic.py \
    --point-cloud '$POINT_CLOUD' \
    --odir '$OUTPUT_DIR' \
    --batch_size $BATCH_SIZE \
    --num_procs $NUM_PROCS \
    --step $STEP \
    --is-wood $IS_WOOD \
    --output_fmt $OUTPUT_FORMAT"

# Add optional arguments if they were provided
if [ ! -z "$TILE_INDEX" ]; then
    CMD="$CMD --tile-index '$TILE_INDEX'"
fi

if [ ! -z "$BUFFER" ]; then
    CMD="$CMD --buffer $BUFFER"
fi

if [ ! -z "$KEEP_NPY" ]; then
    CMD="$CMD $KEEP_NPY"
fi

if [ ! -z "$MODEL" ]; then
    CMD="$CMD --model '$MODEL'"
fi

if [ ! -z "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

# Print the command if verbose
if [ ! -z "$VERBOSE" ]; then
    echo "Executing: $CMD"
fi

# Execute the command
eval $CMD
