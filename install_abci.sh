#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2200
#$ -j y
#$ -o outputs/
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

cd /bb/llm/gaf51275/llm-jp/taishi-work-space/Megatron-DeepSpeed
source .env/bin/activate

pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

python setup.py install
