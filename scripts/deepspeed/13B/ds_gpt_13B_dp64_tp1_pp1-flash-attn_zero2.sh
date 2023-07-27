#!/bin/bash

## GPT-3 13B
model_size=13

num_layers=40
hidden_size=5120
num_attn_heads=40

global_batch_size=1024
lr=1.0e-4
min_lr=1.0e-6
init_std=0.008

sequence_length=2048

### Training duration configs
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
lr_warmup_tokens_in_million=3000
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
mp_size=1 # tensor model parallel size

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=1
no_pp="true"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=2

## Total number of GPUs
num_gpus_pernode=8
num_node=8
num_gpus=$((${num_gpus_pernode} * ${num_node}))

## Data parallel size.
dp_size=$((${num_gpus} / ${pp_size} / ${mp_size}))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=1
###############################################################################
### Misc configs
log_interval=1
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${sequence_length} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=100

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
num_workers=2

# dataset
data_path="dataset/BookCorpusDataset_text_document"
vocab_path="dataset/gpt2-vocab.json"
merge_path="dataset/gpt2-merges.txt"

prescale_grad="true"

# job name
jobname="gpt_${model_size}B_token${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_gpu${num_gpus}"

if [[ $zero_stage -gt 0 ]]; then
  jobname="${jobname}_zero${zero_stage}"
  prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
  jobname="${jobname}_tp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
  jobname="${jobname}_pp${pp_size}"
fi

# output dir
output_home="outputs"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"

## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}

###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
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
    --fp16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --no-async-tensor-model-parallel-allreduce \
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

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"

if [ "${log_optimizer_state}" = "true" ]; then
  megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

# DeepSpeed Config
config_json="scripts/deepspeed/config/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
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

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for ((node = 0; node <= num_node - 1; node++)); do
  if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
    local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
    iteration=$((${local_iteration} > ${iteration} ? ${local_iteration} : ${iteration}))
  fi
done
if [[ $iteration -gt 0 ]]; then
  iteration_2="global_step${iteration}"
  ds_ssh "echo $iteration > $iteration_file"
  ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

# hostfile

source .env/bin/activate

deepspeed --num_nodes ${num_node} \
  --num_gpus ${num_gpus_pernode} \
  --hostfile scripts/deepspeed/hostfile \
  pretrain_gpt.py \
  ${megatron_options} \
  ${data_options} \
  ${deepspeed_options} \
  --wandb-name "deepspeed-flash-attn-${jobname}" \
  &>>${log_path}/${jobname}_${host}_${current_time}.log
