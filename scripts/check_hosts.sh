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
  stat=$(ssh -n $ip which $cmd | grep -v "not found")
  if [ -z "$stat" ]; then
    echo "Host $ip: Warning: $cmd not found. Please install if necessary"
  else
    echo "Host $ip: OK: $cmd exists"
  fi
}

function num_gpus(){
  ip=$1
  expected_slots=$(ssh -n $ip nvidia-smi | grep MiB | grep % | wc -l)
  echo ${expected_slots}
}

function check_host(){
  line="$1"
  # IPアドレスとスロット数を取得
  ip=$(echo $line | cut -d' ' -f1)
  expected_slots=$(echo $line | cut -d'=' -f2)

  # SSHで各ホストに接続し、nvidia-smiコマンドを実行してGPUの数を取得
  gpu_count=$(num_gpus $ip)

  # GPUの数とスロット数を比較
  if [ $gpu_count -eq $expected_slots ]; then
    echo "Host $ip: OK: $expected_slots GPUs found"
  else
    echo "Host $ip: Error - Expected $expected_slots GPUs, but found $gpu_count"
  fi
  # numactl がホストでサポートされているか確認（conda を使わない場合は無視して良い）
  cmd_check $ip numactl
}

# ホストファイルを1行ずつ読み込む
while IFS= read -r line
do
  check_host "$line"
done < "$HOST_FILE"
