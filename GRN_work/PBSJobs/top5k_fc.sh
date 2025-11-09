#! /bin/bash

#PBS -P yr31
#PBS -q gpuvolta
#PBS -l walltime=8:00:00
#PBS -l ncpus=36
#PBS -l mem=32GB
#PBS -l ngpus=3
#PBS -l jobfs=20GB
#PBS -l wd

module load python3/3.9.2
module load pytorch/1.10.0
source  /home/561/rb6232/code_base/Hon_proj_RDB/.venv/bin/activate
python3 /home/561/rb6232/code_base/Hon_proj_RDB/GRN_work/train_FCAE.py --data_path /home/561/rb6232/code_base/data/sct_top5k.npz --num_bins 7 --track_epoch_confusion


