#!/bin/bash

# Usage: ./convert_hdf5.sh <dataset_path> <output_name> [imsize]

# Positional arguments
DATASET_PATH=$1
OUTPUT_NAME=$2
IMSIZE=${3:-84}

# Check if arguments are provided
if [ -z "$DATASET_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: $0 <dataset_path> <output_name>.hdf5"
    exit 1
fi

cd robomimic/robomimic/scripts || { echo "Failed to change directory"; exit 1; }

# Run the Python scripts with the provided arguments
python3 conversion/convert_robosuite.py --dataset "$DATASET_PATH"
python3 split_train_val.py --dataset "$DATASET_PATH" --ratio 0.1
python3 dataset_states_to_obs.py --dataset "$DATASET_PATH" --output_name "$OUTPUT_NAME" --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height "$IMSIZE" --camera_width "$IMSIZE"

# Extract the directory of the input dataset
OUTPUT_DIR=$(dirname "$DATASET_PATH")
FULL_OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"

# Run to show the dataset info
python3 get_dataset_info.py --dataset "$FULL_OUTPUT_PATH"
echo "Conversion completed successfully! Output saved as $FULL_OUTPUT_PATH"