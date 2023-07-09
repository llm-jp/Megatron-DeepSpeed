#!/bin/bash

# ホストファイルのパスを指定
HOST_FILE="$1"

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/check_hosts.sh HOSTFILE"
    exit 1
fi

function cmd_check(){
  ip=$1
  cmd=$2
  stat=$(ssh $ip which $cmd | grep -v "not found")
  if [ -z "$stat" ]; then
    echo "Host $ip: Warning: $cmd not found. Please install if necessary"
  else
    echo "Host $ip: OK: $cmd exists"
  fi
}

function num_gpus(){
  ip=$1
  # expected_slots=$(ssh $ip "nvidia-smi -L | wc -l")
  expected_slots=$(ssh $ip "nvidia-smi | grep MiB | grep % | wc -l")
  echo ${expected_slots}
}

# ホストファイルを1行ずつ読み込む
while read -r line
do
  # IPアドレスとスロット数を取得
  ip=$(echo $line | cut -d' ' -f1)
  expected_slots=$(echo $line | cut -d'=' -f2)

  # SSHで各ホストに接続し、nvidia-smiコマンドを実行してGPUの数を取得
  gpu_count=$(num_gpus $ip)
  # numactl がホストでサポートされているか確認（conda を使わない場合は無視して良い）
  cmd_check $ip numactl

  # GPUの数とスロット数を比較
  if [ $gpu_count -eq $expected_slots ]; then
    echo "Host $ip: OK"
  else
    echo "Host $ip: Error - Expected $expected_slots GPUs, but found $gpu_count"
  fi
done < "$HOST_FILE"
