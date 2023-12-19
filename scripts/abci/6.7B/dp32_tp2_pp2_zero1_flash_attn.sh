#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=10:00:00:00
#$ -j y
#$ -o outputs/llm-jp/6.7B/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

# GPU settings
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# hostfile
mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# GPT-3 6.7B
model_size=6.7

num_layers=32
hidden_size=4096
num_attn_heads=32

global_batch_size=1024

lr=1.2e-4
min_lr=1.0e-6
init_std=0.009

sequence_length=4096

## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=266
train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))

## train_samples is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$((300 * 1000000000 * 2 / ${sequence_length}))

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
lr_warmup_tokens_in_million=2656
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=2 # tensor model parallel size

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=2
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=1

## Data parallel size.
dp_size=$((${NUM_GPUS} / ${pp_size} / ${mp_size}))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/NUM_GPUS
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=4
###############################################################################
### Misc configs
log_interval=1
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=500
estimated_train_iter=$((${train_tokens} / ${sequence_length} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=500

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="true"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
num_workers=0

## Public the Pile dataset, can be downloaded at
## https://mystic.the-eye.eu/public/AI/pile_neox/ or
## https://the-eye.eu/public/AI/pile_neox/ Change data_home to where you
## store the pile_text_document.bin and pile_text_document.idx.

TRAIN_DATASET_PATH="/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train"

data_path=""

# ja wiki
data_path="${data_path} 1489457253 ${TRAIN_DATASET_PATH}/ja_wiki/ja_wiki_merge_1_text_document"
# en wiki
data_path="${data_path} 4983898399 ${TRAIN_DATASET_PATH}/en_wiki/en_wiki_merge_1_text_document"
# code stack
data_path="${data_path} 8967214774 ${TRAIN_DATASET_PATH}/code_stack/code_stack_merge_1_text_document"
# en pile
data_path="${data_path} 17716652494 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_1_text_document"
data_path="${data_path} 17728398911 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_2_text_document"
data_path="${data_path} 17862741217 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_3_text_document"
data_path="${data_path} 17854181202 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_4_text_document"
data_path="${data_path} 17779824310 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_5_text_document"
data_path="${data_path} 17847796716 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_6_text_document"
data_path="${data_path} 8938950206 ${TRAIN_DATASET_PATH}/en_pile/en_pile_merge_7_text_document"
# ja cc
data_path="${data_path} 19540410239 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_1_text_document"
data_path="${data_path} 19559059958 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_2_text_document"
data_path="${data_path} 19547251566 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_3_text_document"
data_path="${data_path} 19550089401 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_4_text_document"
data_path="${data_path} 19553509796 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_5_text_document"
data_path="${data_path} 19566479585 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_6_text_document"
data_path="${data_path} 17060823775 ${TRAIN_DATASET_PATH}/ja_cc/ja_cc_merge_7_text_document"

VALIDATION_DATASET_PATH="/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/val"

VALIDATION_DATA_PATH=""
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 77810430 ${VALIDATION_DATASET_PATH}/code_stack_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 37133061 ${VALIDATION_DATASET_PATH}/en_pile_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1011609 ${VALIDATION_DATASET_PATH}/en_wiki_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 147265562 ${VALIDATION_DATASET_PATH}/ja_cc_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1097003 ${VALIDATION_DATASET_PATH}/ja_wiki_validation_0_text_document"


prescale_grad="true"
jobname="gpt_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${NUM_GPUS}"
if [[ $zero_stage -gt 0 ]]; then
  jobname="${jobname}_z${zero_stage}"
  prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
  jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
  jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}_rebase"

output_home="outputs"
log_path="${output_home}/log/"

checkpoint_path="/groups/gaf51275/llm-jp/checkpoints/megatron-deepspeed/7b/llm-jp-v1.0.1/code10K_en20K_ja_30K/context_4096"

## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --tokenizer-model /bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model \
    --tokenizer-type SentencePieceTokenizer \
    --train-data-path ${data_path} \
    --valid-data-path ${VALIDATION_DATA_PATH} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${sequence_length} \
    --max-position-embeddings ${sequence_length} \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --distributed-backend nccl \
    --bf16 \
    --seed ${seed} \
    --save ${checkpoint_path} \
    --load ${checkpoint_path} \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v2 \
    --use-rotary-position-embeddings \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
  megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
  megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

# DeepSpeed Config
config_json="scripts/deepspeed/config/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="examples_deepspeed/rebase/ds_config_gpt_TEMPLATE_bf16.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} |
  sed "s/MBSIZE/${batch_size}/" |
  sed "s/LOG_INTERVAL/${log_interval}/" |
  sed "s/ZERO_STAGE/${zero_stage}/" |
  sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    >${config_json}

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
  deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
  deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi


mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  ${megatron_options} \
  --use-mpi \
  --wandb-entity "llm-jp" \
  --wandb-project "megatron-lm-7B-2023-1112" \
  --wandb-name "7B-gpt-${jobname}" \
  ${data_options} \
  ${deepspeed_options}