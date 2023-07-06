#!/bin/bash


source /model/share/miniforge/etc/profile.d/conda.sh
conda activate megatron-deepspeed

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH


export NCCL_DEBUG=INFO

export NCCL_DEBUG=INFO
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=CollnetChain


export UCX_DEBUG=info
#printenv >> printenv.log.$OMPI_COMM_WORLD_RANK
	   
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK


case $OMPI_COMM_WORLD_LOCAL_RANK in
        0)      export UCX_NET_DEVICES=mlx5_0:1
                export NCCL_IB_HCA=mlx5_0:1
                ;;
        1)      export UCX_NET_DEVICES=mlx5_1:1
                export NCCL_IB_HCA=mlx5_1:1
                ;;
esac

numactl -l $* --local-rank $OMPI_COMM_WORLD_LOCAL_RANK
