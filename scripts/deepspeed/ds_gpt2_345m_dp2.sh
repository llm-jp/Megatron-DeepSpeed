#!/bin/bash

# Dataset path & checkpoint path
DATASET_PATH=dataset/wiki_ja_20230320_train_filtered_text_document
CHECKPOINT_PATH=checkpoints/gpt2_345m/deepspeed_1node_2gpus
mkdir -p ${CHECKPOINT_PATH}

VOCAB_PATH=/data/tokenizer/v1_20230628/cc100_ja30K_en20K.symbolRemoved.vocab.reestimated.model

# GPT-2 345M (24-layer, 1024-hidden, 16-heads, 345M parameters)
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16

# GPU resources
NUM_NODES=1
NUM_GPUS_PER_NODE=2
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# Parellel parameters
PP_SIZE=1
TP_SIZE=1

DP_SIZE=$((${NUM_GPUS} / (${PP_SIZE} * ${TP_SIZE})))

# Training parameters
GRAD_ACCUMULATION_STEPS=1

MICRO_BATCHSIZE=8
GLOBAL_BATCH_SIZE=$((${MICRO_BATCHSIZE} * ${GRAD_ACCUMULATION_STEPS} * ${DP_SIZE}))

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

TRAINING_ITERATIONS=2
SAVE_INTERVAL=10000
LR_DECAY_ITERATIONS=320000

LR=0.00015
SEED=1234

# deepspeed configuration
CONFIG_FILE=scripts/deepspeed/ds_config_gpt2_345m_dp2.json
ZERO_STAGE=1


# Run Command
deepspeed --num_nodes ${NUM_NODES} \
  --num_gpus ${NUM_GPUS_PER_NODE} \
  pretrain_gpt.py \
  --tokenizer-type 'JapaneseSentencePiece' \
  --tensor-model-parallel-size ${TP_SIZE} \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --micro-batch-size ${MICRO_BATCHSIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
  --train-iters ${TRAINING_ITERATIONS} \
  --save-interval ${SAVE_INTERVAL} \
  --lr-decay-iters ${LR_DECAY_ITERATIONS} \
  --data-path ${DATASET_PATH} \
  --vocab-file ${VOCAB_PATH} \
  --data-impl mmap \
  --split 949,50,1 \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --distributed-backend nccl \
  --override-opt_param-scheduler \
  --lr $LR \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --seed $SEED \
  --no-masked-softmax-fusion \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --deepspeed \
  --deepspeed_config ${CONFIG_FILE} \
  --zero-stage ${ZERO_STAGE} \
  --deepspeed-activation-checkpointing
