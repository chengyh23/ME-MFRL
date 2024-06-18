#!/bin/bash

# REMEMBER: xvfb-run bash

seed=0
indices=(99 149 299 399 499)
for idx in "${indices[@]}"; do
    # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3
    # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs
    # # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs --use_kf_act --kf_proc_model rw
    # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs --use_kf_act --kf_proc_model cv
    # DRL
    python baseline/rl_torch.py --test --idx "$idx" --algo ddpg --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3  --noisy_obs
done