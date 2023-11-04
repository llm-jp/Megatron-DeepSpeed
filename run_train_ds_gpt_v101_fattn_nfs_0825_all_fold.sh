#!/bin/bash -eu


if [ $# -ne 10 ]; then
  echo "Usage: bash $0 [1:ModelSize] [2:GlobalBatch] [3:MicroBatch] [4:ZeROStage] [5:HostFile] [6:NumNodes] [7:TP] [8:PP] [9:NumWorkers] [10:WandbProjName]"
  # sample: bash -x run_train_ds_gpt_v101_fattn_nfs_0825_all_fold.sh 1.3 1536 4 1 hostfile/new-hostfile12 12 1 1 8 perf-test-2023-0822
  exit 1
fi

model_size=$1

# for i in 0 1 2 3 4 5 6 7 8 9 # all
for i in 4 5 6 7 8 9 
# for i in 2 3 # test
do 

echo "Start FOLD 0${i}"

if [ $i = 0 ]; then
  echo -e "Start initial training.\n"
else
  echo -e "Load checkpoint and run training.\n"

  # Set directory name
  ORG_DIR=/data/llmjp0/model_cache_dir/outputs/checkpoint/${model_size}B
  
  pre_fold_num=`expr $i - 1`
  PRE_DIR=${ORG_DIR}/ds_gpt_v101_fattn_nfs_0825_fold-gpt_${model_size}B_fold0${pre_fold_num}_gpu96_node12_lr1.0e-4_gbs1536_mbs1_nwk2_zero1_pp8
  CURRENT_DIR=${ORG_DIR}/ds_gpt_v101_fattn_nfs_0825_fold-gpt_${model_size}B_fold0${i}_gpu96_node12_lr1.0e-4_gbs1536_mbs1_nwk2_zero1_pp8
  echo "PRE_DIR: ${PRE_DIR}"
  echo "CURRENT_DIR: ${CURRENT_DIR}"

  # Prepare checkpoint
  mkdir ${CURRENT_DIR}
  latest_checkpoint_path=${PRE_DIR}/global_step`cat ${PRE_DIR}/latest_checkpointed_iteration.txt`
  echo "latest_checkpoint_path: ${latest_checkpoint_path}"
  ln -s ${latest_checkpoint_path} ${CURRENT_DIR}
  cp ${PRE_DIR}/latest* ${CURRENT_DIR}

fi

CMD="bash run_train_ds_gpt_v101_fattn_nfs_0825_fold.sh $@ 0${i}"

echo "${CMD}"
echo ""
eval "${CMD}"

echo "Cleanup processes."
scripts/cleanup_procs.sh hostfile/new-hostfile12
echo "Sleep 60 sec."
sleep 60
echo "Done"

done
