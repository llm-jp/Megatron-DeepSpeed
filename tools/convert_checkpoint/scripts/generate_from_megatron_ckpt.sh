set -ex
CHECKPOINT_DIR=$1
TOKENIZER_MODEL_PATH=$2


export MASTER_ADDR=localhost
export MASTER_PORT=6000


python ./tests/generate_megatron.py \
       --load ${CHECKPOINT_DIR} \
       --micro-batch-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --no-async-tensor-model-parallel-allreduce \
       --tokenizer-model ${TOKENIZER_MODEL_PATH} \
       --tokenizer-type SentencePieceTokenizer \
       --distributed-backend nccl \
       --use-rotary-position-embeddings \

