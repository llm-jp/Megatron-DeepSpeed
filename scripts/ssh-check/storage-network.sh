#!/bin/bash

# チェックするノードのリスト
nodes=("gpu-node0-storage2" "gpu-node1-storage2" "gpu-node2-storage2" "gpu-node3-storage2" "gpu-node4-storage2" "gpu-node5-storage2" "gpu-node6-storage2" "gpu-node7-storage2" "gpu-node8-storage2" "gpu-node9-storage2" "gpu-node10-storage2" "gpu-node11-storage2" "gpu-node12-storage2" "gpu-node13-storage2" "gpu-node14-storage2" "gpu-node15-storage2")

for node in "${nodes[@]}"; do
  # SSHコマンドを使ってホストに接続します。その結果を/dev/nullにリダイレクトします
  # こうすることで、SSHが成功したか失敗したかを確認できます
  ssh -o BatchMode=yes -o ConnectTimeout=5 $node "echo 2>&1" >/dev/null
  result=$?

  if [ $result -eq 0 ]; then
    echo "SSH to $node - Success"
  else
    echo "SSH to $node - Failed"
  fi
done
