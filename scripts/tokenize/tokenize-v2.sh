#!bash/bash

CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=80

source .env/bin/activate

# Set the output directory:
export OUTDIR=dataset/wikipedia/binarized/v2-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k
mkdir -p $OUTDIR
export MODELDIR=llm-ja-tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.model

# Tokenize and binarize Japanese
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output-prefix $OUTDIR/ja_wiki \
  --vocab-file $MODELDIR \
  --dataset-impl mmap \
  --tokenizer-type JapaneseSentencePiece \
  --workers 64 \
  --append-eod

# Tokenize and binarize English
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/en/en_merged.json \
  --output-prefix $OUTDIR/en_wiki \
  --vocab-file $MODELDIR \
  --dataset-impl mmap \
  --tokenizer-type JapaneseSentencePiece \
  --workers 64 \
  --append-eod
