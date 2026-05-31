#!/usr/bin/env bash
# Pipeline RL launcher.
#
# Usage:
#   bash scale_rl/scripts/train.sh [path/to/train.yaml]
#
# Starts the vLLM rollout worker on VLLM_GPU_ID, then launches the FSDP
# trainer with torchrun.  Both processes share an NCCL rendezvous so the
# trainer can push updated weights into vLLM without checkpointing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/train.yaml}"

# ---------------------------------------------------------------------------
# Parse all needed keys from the yaml in a single python call
# ---------------------------------------------------------------------------
eval "$(python3 - "$CONFIG" <<'EOF'
import sys, re

keys = [
    "model_path", "dtype", "rollout_worker_url", "num_trainer_gpus",
    "vllm_gpu_id", "vllm_gpu_memory_utilization", "vllm_tensor_parallel_size",
    "vllm_weight_transfer_backend", "vllm_clear_kv_cache", "master_port",
]
want = set(keys)
found = {}
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r'^\s*(\w+)\s*:\s*(.+)', line)
        if m and m.group(1) in want:
            found[m.group(1)] = m.group(2).strip().strip('"').strip("'")

for k in keys:
    v = found.get(k, "")
    # export as uppercase shell variable
    print(f'{k.upper()}={v!r}')
EOF
)"

# Extract host and port from the worker URL (http://host:port)
WORKER_HOST="$(echo "$ROLLOUT_WORKER_URL" | sed 's|http://||' | cut -d: -f1)"
WORKER_PORT="$(echo "$ROLLOUT_WORKER_URL" | sed 's|http://||' | cut -d: -f2)"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
VLLM_LOG="${LOG_DIR}/rollout_worker.log"
TRAINER_LOG="${LOG_DIR}/trainer.log"

echo "============================================================"
echo "  Pipeline RL launcher"
echo "  config            : $CONFIG"
echo "  model             : $MODEL_PATH"
echo "  trainer GPUs      : $NUM_TRAINER_GPUS  (CUDA 0...$((NUM_TRAINER_GPUS-1)))"
echo "  vLLM GPU          : $VLLM_GPU_ID"
echo "  rollout worker    : $ROLLOUT_WORKER_URL"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Start the vLLM rollout worker
# ---------------------------------------------------------------------------
echo "[1/3] Starting vLLM rollout worker on GPU $VLLM_GPU_ID ..."

CUDA_VISIBLE_DEVICES="$VLLM_GPU_ID" uv run python3 -m scale_rl.inference.rollout_worker \
    --model        "$MODEL_PATH" \
    --host         "$WORKER_HOST" \
    --port         "$WORKER_PORT" \
    --dtype        "$DTYPE" \
    --gpu-memory-utilization "$VLLM_GPU_MEM" \
    --tensor-parallel-size   "$VLLM_TP_SIZE" \
    --weight-transfer-backend "$VLLM_BACKEND" \
    --clear-kv-cache "$VLLM_CLEAR_KV" \
    >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "    rollout worker PID=$VLLM_PID  log=$VLLM_LOG"

# ---------------------------------------------------------------------------
# 2. Wait for the rollout worker to be healthy
# ---------------------------------------------------------------------------
echo "[2/3] Waiting for rollout worker to become healthy ..."
MAX_WAIT=300
ELAPSED=0
until curl -sf "${ROLLOUT_WORKER_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: rollout worker exited unexpectedly. See $VLLM_LOG"
        exit 1
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: rollout worker did not become healthy within ${MAX_WAIT}s."
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "    rollout worker is healthy."

# ---------------------------------------------------------------------------
# 3. Launch the FSDP trainer with torchrun
#    CUDA_VISIBLE_DEVICES covers only the trainer GPUs (0..N-1).
#    The vLLM worker already owns its physical GPU; the NCCL group
#    addresses it via master_port / rank_offset, not CUDA_VISIBLE_DEVICES.
# ---------------------------------------------------------------------------
TRAINER_GPUS="$(seq -s, 0 $((NUM_TRAINER_GPUS - 1)))"
echo "[3/3] Launching FSDP trainer on GPUs [$TRAINER_GPUS] ..."

cleanup() {
    echo "Shutting down rollout worker (PID=$VLLM_PID) ..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
uv run torchrun \
    --nproc-per-node="$NUM_TRAINER_GPUS" \
    --master-port="$MASTER_PORT" \
    -m scale_rl.scripts.train_entry \
    --config "$CONFIG" \
    2>&1 | tee "$TRAINER_LOG"
