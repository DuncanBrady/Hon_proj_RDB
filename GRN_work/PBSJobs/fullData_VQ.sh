#! /bin/bash

#PBS -P yr31
#PBS -q gpuvolta
#PBS -l walltime=8:00:00
#PBS -l ncpus=24
#PBS -l mem=32GB
#PBS -l ngpus=2
#PBS -l jobfs=100GB
#PBS -l wd

module load python/3.9.2
module load pytorch/1.10.0
source  /home/561/rb6232/code_base/Hon_proj_RDB/.venv/bin/activate
python3 train_vqvae2_binned.py --data-file /g/data/yr31/rb6232/sct_matrix_transposed.npz --out-dir /g/data/yr31/rb6232/fullData_VQ --num-bins 7



