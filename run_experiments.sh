#!/bin/bash

# Array of config files
config_files=(
    "config/code_llama_2_mftcoder_config_cot.yaml"
    "config/code_llama_cot_config.yaml"
    "config/code_llama_2_mftcoder_config_zs.yaml"
    "config/code_llama_config.yaml"
)

# Array of boolean values for load_in_8bit
load_in_8bit_values=("True" "False")

# Loop through each config file
for config_file in "${config_files[@]}"
do
    # Extract a simplified name from the config file path for the output filename
    simplified_name=$(basename "$config_file" .yaml)
    echo "Running experiments for config: $simplified_name"

    # Loop through each load_in_8bit value
    for value in "${load_in_8bit_values[@]}"
    do
        # Check the value of load_in_8bit and set the quantization flag for the output filename accordingly
        if [ "$value" == "True" ]; then
            quantization_flag="8bit"
        else
            quantization_flag="no_8bit"
        fi

        echo "  Testing with load_in_8bit=$value..."

        # Run the experiment with the current config file and load_in_8bit value
        SECOND_CONFIG_LOCATION="$config_file" python main.py arcathon_pipeline.load_in_8bit=$value output_path=./experiments_results/${simplified_name}_${quantization_flag}.csv
    done
    echo "Experiments for $simplified_name completed!"
    echo "--------------------------------------------"
done

echo "All experiments finished!"
