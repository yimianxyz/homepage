#!/bin/bash
# THROUGHPUT GATE — full measurement pass on VM3 (ml-forecast-3, L4 + 4 vCPU).
# Expects the tree extracted at ~/gate (js/, dev/fasteval.js, dev/exact_nn/replica_gate/).
# Usage: nohup bash ~/gate/dev/exact_nn/replica_gate/run_vm3.sh > ~/gate/run.log 2>&1 &
set -x
cd ~/gate/dev/exact_nn/replica_gate
NODE=~/bin/node

echo "=== env ==="
nproc; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$NODE --version; python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"

echo "=== parity (GPU, structure check) ==="
python3 skeleton_torch.py --parity parity_N120_H10.json --device cuda
python3 skeleton_torch.py --parity parity_N120_H90.json --device cuda
python3 skeleton_torch.py --parity parity_N30_H90.json  --device cuda

echo "=== node single-core baseline ==="
$NODE bench_node.js --N 120 --plans 40  | tee node_N120.json
$NODE bench_node.js --N 30  --plans 120 | tee node_N30.json

echo "=== GPU skeleton, CUDA graphs ==="
python3 skeleton_torch.py --device cuda --graphs 1 --Ns 120,30 \
  --Bs 256,1024,4096,16384 --json gpu_graphs.json

echo "=== GPU skeleton, eager (reference) ==="
python3 skeleton_torch.py --device cuda --graphs 0 --Ns 120,30 \
  --Bs 256,4096 --target-sec 5 --json gpu_eager.json

echo "=== DONE ==="
