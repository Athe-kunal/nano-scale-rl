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

scalar_keys = [
    "model_path", "dtype", "rollout_worker_url",
    "vllm_gpu_memory_utilization",
    "vllm_weight_transfer_backend", "vllm_clear_kv_cache", "master_port",
]
want = set(scalar_keys) | {"trainer_gpu_ids", "vllm_gpu_ids"}
found = {}
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r'^\s*(\w+)\s*:\s*(.+)', line)
        if m and m.group(1) in want:
            found[m.group(1)] = m.group(2).strip().strip('"').strip("'")

for k in scalar_keys:
    v = found.get(k, "")
    print(f'{k.upper()}={v!r}')

# Parse trainer_gpu_ids: accepts "[0, 1, 2]" or "0, 1, 2"
raw = found.get("trainer_gpu_ids", "0")
ids = re.findall(r'\d+', raw)
print(f'TRAINER_GPU_IDS={",".join(ids)!r}')
print(f'NUM_TRAINER_GPUS={len(ids)!r}')

# Parse vllm_gpu_ids: tensor_parallel_size = number of GPUs
raw = found.get("vllm_gpu_ids", "0")
vids = re.findall(r'\d+', raw)
print(f'VLLM_GPU_IDS={",".join(vids)!r}')
print(f'VLLM_TENSOR_PARALLEL_SIZE={len(vids)!r}')
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
echo "  trainer GPUs      : [$TRAINER_GPU_IDS]  (${NUM_TRAINER_GPUS} GPUs)"
echo "  vLLM GPU          : $VLLM_GPU_IDS"
echo "  rollout worker    : $ROLLOUT_WORKER_URL"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Start the vLLM rollout worker
# ---------------------------------------------------------------------------
echo "[1/3] Starting vLLM rollout worker on GPU $VLLM_GPU_IDS ..."

CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS" setsid uv run python3 -m scale_rl.inference.rollout_worker \
    --model        "$MODEL_PATH" \
    --host         "$WORKER_HOST" \
    --port         "$WORKER_PORT" \
    --dtype        "$DTYPE" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size   "$VLLM_TENSOR_PARALLEL_SIZE" \
    --weight-transfer-backend "$VLLM_WEIGHT_TRANSFER_BACKEND" \
    --clear-kv-cache "$VLLM_CLEAR_KV_CACHE" \
    >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "    rollout worker PID=$VLLM_PID  log=$VLLM_LOG"

# Register cleanup immediately after the worker starts so any subsequent
# failure (health-check timeout, torchrun error, Ctrl-C) still kills it.
cleanup() {
    echo "Shutting down rollout worker process group (PID=$VLLM_PID) ..."
    # setsid was used when starting the worker so it leads its own process group.
    # Sending SIGTERM to the negative PID kills every process in that group
    # (the root worker + all vLLM/CUDA child processes it spawned).
    kill -TERM -"$VLLM_PID" 2>/dev/null || true
    # Wait up to 10 s for a clean exit, then SIGKILL the whole group.
    local deadline=$(( $(date +%s) + 10 ))
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "Force-killing rollout worker process group ..."
            kill -KILL -"$VLLM_PID" 2>/dev/null || true
            break
        fi
        sleep 0.5
    done
    echo "Rollout worker stopped."
}
trap cleanup EXIT INT TERM HUP

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
echo "[3/3] Launching FSDP trainer on GPUs [$TRAINER_GPU_IDS] ..."

PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$TRAINER_GPU_IDS" \
uv run torchrun \
    --nproc-per-node="$NUM_TRAINER_GPUS" \
    --master-port="$MASTER_PORT" \
    -m scale_rl.scripts.train_entry \
    --config "$CONFIG" \
    2>&1 | tee "$TRAINER_LOG"
