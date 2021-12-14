#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

GPUS=4
CPUS=6
EXP_NAME=restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_anime
QUEUE_NAME=
NODE=57
EXP_PATH=realesrgan_c64b23g32_12x4_lr1e-4_400k_anime_selfie
echo GLOG_vmodule=MemcachedClient=-1 srun -p sgres --mpi=pmi2 --job-name=$QUEUE_NAME --gres=gpu:$GPUS --ntasks=$GPUS --ntasks-per-node=$GPUS --cpus-per-task=$CPUS --kill-on-bad-exit=1 --begin=now+0 -w ABUD-IDC1-10-56-2-$NODE python -u tools/train.py configs/$EXP_NAME.py --work-dir=./experiments/$EXP_PATH --seed=0 --launcher="slurm"

GLOG_vmodule=MemcachedClient=-1 srun -p sgres --mpi=pmi2 --job-name=$QUEUE_NAME --gres=gpu:$GPUS --ntasks=$GPUS --ntasks-per-node=$GPUS --cpus-per-task=$CPUS --kill-on-bad-exit=1 --begin=now+0 -w ABUD-IDC1-10-56-2-$NODE python -u tools/train.py configs/$EXP_NAME.py --work-dir=./experiments/$EXP_PATH --seed=0 --launcher="slurm"

