#!/bin/bash

# Initialize Conda for script use
eval "$(conda shell.bash hook)"

# Function to run a Python script within a Conda environment
run_python_script() {
    local env_name=$1
    local script_name=$2
    local script_args=$3
    mkdir -p ./logs
    local output_file="./logs/output_${script_name%.*}_${script_args}.log"
    
    echo "Activating $env_name and running $script_name"
    conda activate "$env_name"
    # output=$(python3 "$script_name" "$script_args" 2>&1)
    python3 "$script_name" "$script_args" > "$output_file" 2>&1
    status=$?

    if [ $status -ne 0 ]; then
        echo "$script_name failed with output:"
        # echo "$output"
        exit $status
    fi
    conda deactivate
    pkill python
    pkill python3
}

for i in {7..16}; do
    echo "Running iteration: $i"
    run_python_script "vllm" "vllm_gen.py" "$i"
    run_python_script "torch" "train_value.py" "$i"
    run_python_script "torch" "train_policy.py" "$i"
done

echo "All scripts executed successfully for all iterations."