#!/bin/bash

# Check if an algorithm name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <algorithm_name>"
    exit 1
fi

algo=$1

# List of seeds
seeds=(3 7)
# seeds=(0 3 7 11 13)
# seeds=(0 3 7 11 13 15 18 20 32 42)

# Create a temporary copy of train_my.py
# to ensure that it is always executing the intended python code
project_dir="/home/chengyh23/Documents/ME-MFRL"
temp_dir=$(mktemp -d "/home/chengyh23/Documents/.tmp/ME-MFRL.XXXXXX")
# rsync -av --exclude-from="$project_dir/.gitignore" --exclude='.tmp/' "$project_dir/" "$temp_dir"
rsync -av --exclude-from="$project_dir/.gitignore" "$project_dir/" "$temp_dir"
# temp_script=$(mktemp .tmp/train_my.XXXXXX.py)
# cp train_my.py "$temp_script"


# Function to clean up the temporary script
cleanup() {
    # rm -f "$temp_script"
    rm -rf "$temp_dir"
}
trap cleanup EXIT

# Loop through the seeds and run the training script
for seed in "${seeds[@]}"; do
    # python "$temp_dir/train_my.py" --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --use_wandb True
    python "$temp_dir/train_my.py" --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --num_adversaries 30 --num_good_agents 10 --noisy_obs --use_kf_act --kf_proc_model cv --use_wandb
    python "$temp_dir/train_my.py" --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --num_adversaries 30 --num_good_agents 10 --noisy_obs --use_kf_act --kf_proc_model rw --use_wandb
    # python "$temp_dir/train_my.py" --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --num_adversaries 30 --num_good_agents 10 --noisy_obs --use_wandb
    # python "$temp_dir/train_my.py" --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --num_adversaries 20 --num_good_agents 20 --use_wandb
    # python train_my.py --algo "$algo" --n_round 500 --max_steps 400 --seed "$seed" --use_wandb True
done