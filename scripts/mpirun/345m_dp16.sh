#! /bin/bash

# Runs the "345M" parameter model

# distributed settings
GPUS_PER_NODE=8
NNODES=8
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# load virtualenv
source /model/hpc-team/Megatron-DeepSpeed/.env/bin/activate

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_345m/2node-16gpu-mpirun

mkdir -p $CHECKPOINT_PATH

HOSTFILE=hostfile
if [ ! -e "$HOSTFILE" ]; then
  $echo "Error: Hostfile '$HOSTFILE' not found"
  exit 1
fi


# Open MPI training

mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -hostfile ${HOSTFILE} \
  -x MASTER_ADDR=10.2.72.135 \
  -x MASTER_PORT=16500 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO  -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --micro-batch-size 4 \
  --global-batch-size 512 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --data-path ${DATA_PATH} \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 10000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --use-mpi \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "gpt2_345m_2node_dp16-mpirun"
