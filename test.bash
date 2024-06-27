#!/bin/bash

# REMEMBER: xvfb-run bash

# seed=0
# indices=(99 149 299 399 499)
# for idx in "${indices[@]}"; do
#     # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3
#     # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs
#     # # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs --use_kf_act --kf_proc_model rw
#     # python train_my.py --test --render --render_every 30 --idx "$idx" --algo dqn --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3 --noisy_obs --use_kf_act --kf_proc_model cv
#     # DRL
#     python baseline/rl_torch.py --test --idx "$idx" --algo ddpg --n_round 500 --max_steps 400  --seed "$seed" --num_adversaries 3  --noisy_obs

# done

# Ns=(4 6)
# for N in "${Ns[@]}"; do
#     python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries "$N" --noisy_obs --use_kf_act --kf_proc_model cv --idx 4999
#     python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries "$N" --noisy_obs --idx 4999
#     python baseline/janosov.py --algo janosov --test --test_n_round 100 --num_adversaries "$N" --noisy_obs
# done

NFs=(2 4)
for NF in "${NFs[@]}"; do
    python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor "$NF" --use_kf_act --kf_proc_model cv --idx 4999
    # python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor "$NF" --idx 4999
    # python baseline/janosov.py --algo janosov --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor "$NF"
done

# python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --eps_k 0.4 --noisy_obs --use_kf_act --kf_proc_model cv --idx 4999
# python test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --eps_k 0.4 --noisy_obs  --idx 4999