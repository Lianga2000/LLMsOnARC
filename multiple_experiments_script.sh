#!/bin/bash

# Define the possible options for CONFIG_NAMES
options1=("cot_config")
options2=("code_llama_7b_config" "code_llama_13b_config" "mixtral_config" "phind_mftcoder_config")

# Define the possible values for the boolean flags
using_numbers_values=("True")
load_in_8bit_values=("True")

# Iterate over all the combinations
for option1 in "${options1[@]}"; do
    for option2 in "${options2[@]}"; do
        for using_numbers in "${using_numbers_values[@]}"; do
            for load_in_8bit in "${load_in_8bit_values[@]}"; do
                # Construct the CONFIG_NAMES part
                if [ -z "$option1" ]; then
                    config_names="CONFIG_NAMES=${option2}"
                else
                    config_names="CONFIG_NAMES=${option1},${option2}"
                fi

                # Run the experiment command
                command="${config_names} python main.py using_numbers=${using_numbers} arcathon_pipeline.args.load_in_8bit=${load_in_8bit}"
                echo "Running command: $command"
                eval $command
            done
        done
    done
done

echo "All experiments have been run."
