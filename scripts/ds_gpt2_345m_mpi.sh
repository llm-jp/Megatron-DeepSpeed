#! /bin/bash

# Runs the "345M" parameter model

# Load conda environment (TODO: to be replaced with pyenv)
source /model/share/miniforge/etc/profile.d/conda.sh
conda activate megatron-deepspeed
# source /model/hpc-team/Megatron-DeepSpeed/.env/bin/activate

export OMP_NUM_THREADS=9


# Initialize variables
NUM_NODES=""
NUM_GPUS_PER_NODE=""
HOSTFILE=""
PP_SIZE=""
TP_SIZE=""

# Function to display usage
usage() {
  echo "Usage: $0 [-n|--nodes <number of nodes>] [-g|--gpus <number of GPUs per node>] [-f|--hostfile <hostfile path>] [--pp <Pipeline parallel size>] [--tp <Tensor parallel size>]"
  exit 1
}


# Parse command line options
while (( "$#" )); do
  case "$1" in
    -h|--help)
      usage
      ;;
    -n|--nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    -g|--gpus)
      NUM_GPUS_PER_NODE="$2"
      shift 2
      ;;
    -f|--hostfile)
      HOSTFILE="$2"
      shift 2
      ;;
    --pp)
      PP_SIZE="$2"
      shift 2
      ;;
    --tp)
      TP_SIZE="$2"
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# Check if all parameters are set
if [ -z "$NUM_NODES" ] || [ -z "$NUM_GPUS_PER_NODE" ] || [ -z "$HOSTFILE" ] || [ -z "$PP_SIZE" ] || [ -z "$TP_SIZE" ]; then
    echo "Error: Not all parameters are set. Please check your input."
    usage
fi

if [ ! -e "$HOSTFILE" ]; then
  $echo "Error: Hostfile '$HOSTFILE' not found"
  exit 1
fi


TOTAL_NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
NUMACTL_SETUP_SCRIPT=./scripts/numactl/setup_${NUM_GPUS_PER_NODE}.sh

DP_SIZE=$((${TOTAL_NUM_GPUS} / (${PP_SIZE} * ${TP_SIZE})))

# Print the input parameters
echo "Number of nodes: $NUM_NODES"
echo "Number of GPUs per node: $NUM_GPUS_PER_NODE"
echo "Number of total GPUs: $TOTAL_NUM_GPUS"
echo "Hostfile path: $HOSTFILE"
echo "numactl setup script: $NUMACTL_SETUP_SCRIPT"
echo "Pipeline parallel size: $PP_SIZE"
echo "Tensor parallel size: $TP_SIZE"
echo "Data parallel size: $DP_SIZE"


# Dataset path & checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_345m/ds_${NUM_GPUS_PER_NODE}gpu
mkdir -p ${CHECKPOINT_PATH}

VOCAB_PATH=dataset/gpt2-vocab.json
MERGE_PATH=dataset/gpt2-merges.txt

# GPT-2 345M (24-layer, 1024-hidden, 16-heads, 345M parameters)
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16


# Training parameters
GRAD_ACCUMULATION_STEPS=1

MICRO_BATCHSIZE=4  # should be less than 8 for many VMs
GLOBAL_BATCH_SIZE=$((MICRO_BATCHSIZE * DP_SIZE))

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

TRAINING_ITERATIONS=10
SAVE_INTERVAL=10
LR_DECAY_ITERATIONS=320000

LR=0.00015
LR_WARMUP_ITER=32000
SEED=1234

# deepspeed configuration
CONFIG_FILE=scripts/ds_config_gpt2_345m.json
ZERO_STAGE=1

# for debug
export CUDA_LAUNCH_BLOCKING=1

mpiexec -n $TOTAL_NUM_GPUS -npernode $NUM_GPUS_PER_NODE -machinefile "$HOSTFILE" \
	-x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS --report-bindings --map-by slot:PE=$OMP_NUM_THREADS --bind-to core  \
	"$NUMACTL_SETUP_SCRIPT" python pretrain_gpt.py \
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
  --data-path ${DATA_PATH} \
  --vocab-file ${VOCAB_PATH} \
  --merge-file ${MERGE_PATH} \
  --data-impl mmap \
  --split 92,6,2 \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --distributed-backend nccl \
  --override-lr-scheduler \
  --lr $LR \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-iters $LR_WARMUP_ITER \
  --checkpoint-activations \
  --log-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --fp16 \
  --seed $SEED \
  --no-masked-softmax-fusion \
  --deepspeed \
  --deepspeed_config ${CONFIG_FILE} \
  --zero-stage ${ZERO_STAGE} \
  --deepspeed-activation-checkpointing \
  --wandb-name "gpt2_345m_${NUM_NODES}node_dp${TOTAL_NUM_GPUS}-mpirun"
