#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/tokenize/
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

# set the input directory
INPUT_DIR=/bb/llm/gaf51275/llm-jp/datasets/llm-jp-corpus/108b

# set the output directory
OUTPUT_DIR=/bb/llm/gaf51275/llm-jp/datasets/binarized/v1.0.2/code20K_en40K_ja60K.ver2.2
mkdir -p $OUTPUT_DIR

# set the model directory
MODEL_DIR=/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2
MODEL_PATH=$MODEL_DIR/code20K_en40K_ja60K.ver2.2.model

# Tokenize and binarize

# refined en-wiki & en-wiki
for file in $INPUT_DIR/*en_wiki*; do
  if [ -f "$file" ]; then
    python tools/preprocess_data.py \
      --input "$file" \
      --output-prefix "$OUTPUT_DIR/$(basename "$file" .${file##*.})" \
      --tokenizer-model "$MODEL_PATH" \
      --dataset-impl mmap \
      --tokenizer-type SentencePieceTokenizer \
      --workers 64 \
      --append-eod
  fi
done

