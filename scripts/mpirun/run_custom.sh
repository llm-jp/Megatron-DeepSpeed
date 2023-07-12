#! /bin/bash

# Runs the GPT2-based parameter model

# Initialize variables
NNODES=""
GPUS_PER_NODE=""
HOSTFILE=""
PP_SIZE=""
TP_SIZE=""
MODEL_SIZE=""

# Function to display usage
usage() {
  echo "Usage: $0 [-n|--nodes <number of nodes>] [-g|--gpus <number of GPUs per node>] [-f|--hostfile <hostfile path>] [-m|--model <model size>] [--pp <Pipeline parallel size>] [--tp <Tensor parallel size>]"
  exit 1
}

# Parse command line options
while (( "$#" )); do
  case "$1" in
    -h|--help)
      usage
      ;;
    -n|--nodes)
      NNODES="$2"
      shift 2
      ;;
    -g|--gpus)
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    -f|--hostfile)
      HOSTFILE="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_SIZE="$2"
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
if [ -z "$NNODES" ] || [ -z "$GPUS_PER_NODE" ] || [ -z "$HOSTFILE" ] || [ -z "$MODEL_SIZE" ] || [ -z "$PP_SIZE" ] || [ -z "$TP_SIZE" ]; then
    echo "Error: Not all parameters are set. Please check your input."
    usage
fi

if [ ! -e "$HOSTFILE" ]; then
  $echo "Error: Hostfile '$HOSTFILE' not found"
  exit 1
fi

case "$MODEL_SIZE" in
  350m)
    NUM_LAYERS=24
    HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=16
    ;;
  760m)
    NUM_LAYERS=24
    HIDDEN_SIZE=1536
    NUM_ATTN_HEADS=16
    ;;
  800m)
    NUM_LAYERS=16
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=8
    ;;
  1.3b)
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    ;;
  2.7b)
    NUM_LAYERS=32
    HIDDEN_SIZE=2560
    NUM_ATTN_HEADS=32
    ;;
  6.7b)
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    ;;
  13b)
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    ;;
  20b)
    NUM_LAYERS=44
    HIDDEN_SIZE=6144
    NUM_ATTN_HEADS=64
    ;;
  26b)
    NUM_LAYERS=48
    HIDDEN_SIZE=6144
    NUM_ATTN_HEADS=64
    ;;
  *)
    echo "Error: Unsupported model size $MODEL_SIZE, must be 350m, 760m, 800m, 1.3b, 2.7b, 6.7b, 13b, 20b or 26b"
    exit 1
    ;;
esac


WORLD_SIZE=$((${NNODES} * ${GPUS_PER_NODE}))
NUMACTL_SETUP_SCRIPT=./scripts/numactl/setup_${GPUS_PER_NODE}.sh

DP_SIZE=$((${WORLD_SIZE} / (${PP_SIZE} * ${TP_SIZE})))

# Print the input parameters
echo "Number of nodes: $NNODES"
echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Number of total GPUs: $WORLD_SIZE"
echo "Hostfile path: $HOSTFILE"
echo "numactl setup script: $NUMACTL_SETUP_SCRIPT"
echo "Pipeline parallel size: $PP_SIZE"
echo "Tensor parallel size: $TP_SIZE"
echo "Data parallel size: $DP_SIZE"
echo "Number of layers: $NUM_LAYERS"
echo "Hidden layer size: $HIDDEN_SIZE"
echo "Number of attention heads: $NUM_ATTN_HEADS"

# load virtualenv
source /model/hpc-team/Megatron-DeepSpeed/.env/bin/activate

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH="checkpoints/gpt2_${MODEL_SIZE}/${NNODES}node-${WORLD_SIZE}gpu-mpirun"

rm -rf "$CHECKPOINT_PATH"
mkdir -p "$CHECKPOINT_PATH"


# Open MPI training
RUN_NAME="gpt2_${MODEL_SIZE}_${NNODES}n_${WORLD_SIZE}g-${NUM_LAYERS}_${HIDDEN_SIZE}_${NUM_ATTN_HEADS}-dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun"
DATETIME=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs
OUT_LOG="logs/${RUN_NAME}_${DATETIME}.log"

echo "Name: ${RUN_NAME}"
echo "Output log: ${OUT_LOG}"

MASTER_ADDR=$(head -n 1 $HOSTFILE | cut -d' ' -f1)

mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -hostfile ${HOSTFILE} \
  -x MASTER_ADDR="$MASTER_ADDR" \
  -x MASTER_PORT=16500 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO  -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTN_HEADS \
  --micro-batch-size 4 \
  --global-batch-size 512 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 10 \
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
  --wandb-name "${RUN_NAME}" 2>&1 | tee "${OUT_LOG}"
