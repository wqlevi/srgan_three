#!/bin/bash -l
##standard output and error:
#SBATCH -o /u/wangqi/log/srgan_fe.out.%j
#SBATCH -e /u/wangqi/log/srgan_fe.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J TORCH-GPU
# Node feature
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:rtx_5000:4
# Number of nodes and MPI tasks per node:
# wall clock limit(Max. is 24hrs)
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2021.11
### module load cuda/11.4
# pytorch
module load pytorch/gpu-cuda-11.4/1.12.0
python /u/wangqi/git_wq/srgan_three/train.py --data_path '/u/wangqi/Celeba/train' --arch_type 'VGG19' >> out.txt 2>>err.txt

