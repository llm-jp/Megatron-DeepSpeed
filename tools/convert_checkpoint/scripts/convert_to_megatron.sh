set -ex
CHECKPOINT_DIR=$1
OUTPUT_DIR=$2

python deepspeed_to_megatron.py \
       --input_folder ${CHECKPOINT_DIR} \
       --output_folder ${OUTPUT_DIR} \
       --target_tp 1 \
       --target_pp 1

