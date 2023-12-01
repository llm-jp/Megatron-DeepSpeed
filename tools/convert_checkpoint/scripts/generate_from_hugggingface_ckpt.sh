set -ex
CHECKPOINT_DIR=$1

python ./tests/generate_huggingface.py \
       --model_name_or_path ${CHECKPOINT_DIR} \
       ${@:2}

