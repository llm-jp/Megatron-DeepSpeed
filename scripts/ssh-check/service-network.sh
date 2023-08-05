#!/bin/bash

# チェックするノードのリスト
nodes=("gpu-node0" "gpu-node1" "gpu-node2" "gpu-node3" "gpu-node4" "gpu-node5" "gpu-node6" "gpu-node7" "gpu-node8" "gpu-node9" "gpu-node10" "gpu-node11" "gpu-node12" "gpu-node13" "gpu-node14" "gpu-node15")

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
