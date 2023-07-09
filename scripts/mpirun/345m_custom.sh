#! /bin/bash

# Runs the "345M" parameter model

# Initialize variables
NNODES=""
GPUS_PER_NODE=""
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
if [ -z "$NNODES" ] || [ -z "$GPUS_PER_NODE" ] || [ -z "$HOSTFILE" ] || [ -z "$PP_SIZE" ] || [ -z "$TP_SIZE" ]; then
    echo "Error: Not all parameters are set. Please check your input."
    usage
fi

if [ ! -e "$HOSTFILE" ]; then
  $echo "Error: Hostfile '$HOSTFILE' not found"
  exit 1
fi


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

# --train-iters 500000 \

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
  --train-iters 100 \
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
  --wandb-name "gpt2_345m_${NNODES}node_dp${DP_SIZE}-mpirun"
