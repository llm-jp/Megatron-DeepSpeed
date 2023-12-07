#!/bin/bash
#$ -l rt_AF=49
#$ -l h_rt=50:00:00:00
#$ -j y
#$ -o outputs/175B/49node/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llm-jp/Megatron-DeepSpeed
source .env/bin/activate

# GPT-3 175B
model_size=175

num_layers=96
hidden_size=12288
num_attn_heads=96

global_batch_size=1536

lr=0.6e-4
min_lr=1.0e-6
init_std=0.005

sequence_length=2048

# LLM-jp 175B parameter model (ABCI 2023.10/5-12/4)
train_tokens_in_billion=108
train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))

## train_samples is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$((108 * 1000000000 * 2 / ${sequence_length}))

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000

###############################################################################
# lr configs
lr_warmup_tokens_in_million=10800
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="constant"
###############################################################################
# Tensor Model Parallel
mp_size=4
# Pipeline parallelism
pp_size=49
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=1

## Total number of GPUs
num_gpus_pernode=8
num_node=$NHOSTS
num_gpus=$((${num_gpus_pernode} * ${num_node}))

## Data parallel size.
dp_size=$((${num_gpus} / ${pp_size} / ${mp_size}))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=2
###############################################################################
### Misc configs
log_interval=1

eval_iters=10
eval_interval=100

estimated_train_iter=$((${train_tokens} / ${sequence_length} / ${global_batch_size}))
save_interval=250

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
num_workers=1

# train dataset
DATASET_PATH="/bb/llm/gaf51275/llm-jp/datasets/binarized/v1.0.2/code20K_en40K_ja60K.ver2.2"

DATA_PATH=""

# ja wiki
DATA_PATH="${DATA_PATH} 2063613609 $DATASET_PATH/ja_wiki_text_document"
# ja cc
DATA_PATH="${DATA_PATH} 51268342277 $DATASET_PATH/ja_cc_text_document"
# en wiki
DATA_PATH="${DATA_PATH} 6205632984 $DATASET_PATH/en_wiki_text_document"
# en pile
DATA_PATH="${DATA_PATH} 39710501236 $DATASET_PATH/en_pile_text_document"
# code stack
DATA_PATH="${DATA_PATH} 3323268568 $DATASET_PATH/code_stack_text_document"

data_path=$DATA_PATH

# validation dataset
VALIDATION_DATASET="/bb/llm/gaf51275/llm-jp/datasets/binarized/v1.0.2/code20K_en40K_ja60K.ver2.2/validation"
VALIDATION_DATA_PATH="${VALIDATION_DATASET}/validation_text_document"

# job name setting
prescale_grad="true"
jobname="gpt_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
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
checkpoint_path="/bb/llm/gaf51275/llm-jp/checkpoints/megatron-deepspeed/175B/ABCI/49node/${jobname}-flash-attn-rope_bf16-vocab_100K"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --tokenizer-model /bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code20K_en40K_ja60K.ver2.2.model \
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
    --pp-partition-method type:transformer|embedding \
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
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --use-flash-attn \
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

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=8"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

mpirun -np $num_gpus \
  --npernode $num_gpus_pernode \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  ${megatron_options} \
  --use-mpi \
  --wandb-name "175B-100K-${jobname}" \
  ${data_options} \
  ${deepspeed_options}
