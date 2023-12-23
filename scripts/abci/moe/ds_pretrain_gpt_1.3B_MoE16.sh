#!/bin/bash
#$ -l rt_AF=2
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/125M_MoE8/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llm-jp/taishi-work-space/Megatron-DeepSpeed
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

## GPT-3 1.3B
model_size=1.3

num_layers=24
hidden_size=2048
num_attn_heads=16

global_batch_size=512
lr=1.2e-4
min_lr=1.0e-6
init_std=0.014

sequence_length=2048

## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=300
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
# LR_DECAY_TOKENS=260000000000

lr_warmup_tokens_in_million=375
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=300000000000
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1 # tensor model parallel size

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=1
no_pp="true"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=1

## Data parallel size.
dp_size=$((${NUM_GPUS} / ${pp_size} / ${mp_size}))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/NUM_GPUS
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=1
# EP_SIZE=64
EP_SIZE="16"

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi


## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
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
activation_checkpoint="false"

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

checkpoint_path="/bb/llm/gaf51275/llm-jp/taishi-work-space/checkpoints/megatron-deepspeed/${jobname}_EP_PARALLEL_SIZE_${EP_PARALLEL_SIZE}_EP_SIZE_${EP_SIZE}"

## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
DATASET_DIR=/bb/llm/gaf51275/llm-jp/taishi-work-space/binarized/ver2.2/code10K_en20K_ja30K/

TRAIN_DATA_PATH=""
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1489457253 ${DATASET_DIR}/ja_wiki/ja_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983898399 ${DATASET_DIR}/en_wiki/en_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8967214774 ${DATASET_DIR}/code_stack/code_stack_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17716652494 ${DATASET_DIR}/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17728398911 ${DATASET_DIR}/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17862741217 ${DATASET_DIR}/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17854181202 ${DATASET_DIR}/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17779824310 ${DATASET_DIR}/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17847796716 ${DATASET_DIR}/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8938950206 ${DATASET_DIR}/en_pile/en_pile_merge_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19540410239 ${DATASET_DIR}/ja_cc/ja_cc_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19559059958 ${DATASET_DIR}/ja_cc/ja_cc_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19547251566 ${DATASET_DIR}/ja_cc/ja_cc_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19550089401 ${DATASET_DIR}/ja_cc/ja_cc_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19553509796 ${DATASET_DIR}/ja_cc/ja_cc_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19566479585 ${DATASET_DIR}/ja_cc/ja_cc_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17060823775 ${DATASET_DIR}/ja_cc/ja_cc_merge_7_text_document"

VALID_DATA_PATH=""

VALID_DATA_PATH="${VALID_DATA_PATH} 77810430 ${DATASET_DIR}/val/code_stack_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 37133061 ${DATASET_DIR}/val/en_pile_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 1011609 ${DATASET_DIR}/val/en_wiki_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 147265562 ${DATASET_DIR}/val/ja_cc_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 1097003 ${DATASET_DIR}/val/ja_wiki_validation_0_text_document"


data_options=" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /bb/llm/gaf51275/llm-jp/taishi-work-space/llm-jp-tokenizer/models/ver2.2/code10K_en20K_ja30K.ver2.2.model \
    --train-data-path ${TRAIN_DATA_PATH} \
    --valid-data-path ${VALID_DATA_PATH} \
    --test-data-path ${VALID_DATA_PATH} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
    --num-experts ${EP_SIZE} \
    --moe-loss-coeff ${MLC} \
    --mlp-type residual \
    --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
    --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
    --moe-min-capacity ${MOE_MIN_CAP} \
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
    --lr-decay-style cosine \
    --split 949,50,1 \
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
    --use-flash-attn-v2 \
    --tensorboard-queue-size 1
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
  megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

megatron_options="${megatron_options} \
        --create-moe-param-group"


if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
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
  -x NCCL_DEBUG=INFO  -x PATH \
  python pretrain_gpt.py \
  ${megatron_options} \
  --use-mpi \
  --wandb-entity "llm-jp" \
  --wandb-project "megatron-deepspeed-moe" \
  --wandb-name "moe_EP_SIZE_${EP_SIZE}_EP_PARALLEL_SIZE_${EP_PARALLEL_SIZE}-${jobname}" \
  ${data_options} \
  ${deepspeed_options}