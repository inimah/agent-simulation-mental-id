#!/bin/bash

# ── Conda ──────────────────────────────────────────────────────────────────────
source CONDA_HOME
conda activate env-name

# run_psydial.sh — PsyDial-inspired (arXiv:2408.15787) RMRR baseline.
#
# Adapts PsyDial's four Retrieve-Mask-Reconstruct-Refine principles for
# Indonesian generation-from-scratch (not reconstructing real sessions):
#
#   Retrieve    — scenario Deskripsi seeds the role card chief_complaint.
#   Mask        — unknown demographics (gender, age, family relationships)
#                 use [TIDAK DIKETAHUI] placeholders; model reveals them
#                 gradually through conversation.
#   Reconstruct — client-first alternating loop (PsyDial Algorithm 1).
#   Refine      — counselor turns that are >50 words, contain newlines, or
#                 use numbered lists are rewritten by a second Ollama call.
#
# LDD metric (Lexical Diversity per Dialogue = PUW × UWPD) is computed
# per-dialogue (stored in each output JSON) and corpus-wide at run end,
# enabling apple-to-apple comparison with published PsyDial statistics.
#
# Default --turns 10 (10 turn-pairs ≈ 20 utterances per dialogue).
# Work list: 155 unique (scenario, emotion, agent_type) triples from
#   data/simulated-sft-fin/ — matches sft-fin reference corpus exactly.
#
# Big-Five traits and resistance are constrained per agent_type (same as v5):
#   verbose    → Ekstroversi/Keramahan: sedang–tinggi, Neurotisisme: rendah–sedang
#   pendiam    → Ekstroversi: rendah, Neurotisisme: sedang–tinggi
#   menghindar → Keterbukaan: rendah, Neurotisisme: tinggi
#   defensif   → Keramahan: rendah, Neurotisisme: tinggi, Resistensi: tinggi
#
# Output dir: data/simulated-sft_baseline_b_psydial
# Output filenames: baseline_b_psydial_{scenario_id:04d}_{emotion}_{agent_type}_{run_ts}.json
#
# Usage:
#   bash run_psydial.sh [extra args passed to simulate_baseline_b_psydial.py]
#   bash run_psydial.sh --resume
#   bash run_psydial.sh --scenario-ids 1 2 3
#   bash run_psydial.sh --turns 20
#   bash run_psydial.sh --dry-run
#   bash run_psydial.sh --manifest-dir /path/to/other/reference/dir

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-PROJECT_ROOT/PsyDial}"

# ── Ollama server — start on this compute node ────────────────────────────────
export OLLAMA_HOST="http://localhost:12434"
export OLLAMA_MODELS="$HOME/.ollama/models"
export PATH="$HOME/.local/ollama/bin:$PATH"
OLLAMA_BIN="$HOME/.local/ollama/bin/ollama"

# Unset CUDA_VISIBLE_DEVICES so Ollama can enumerate the GPU itself.
# SLURM sets this to restrict the job to one GPU, but Ollama's internal
# CUDA probe mismatches against an externally pre-set value and silently
# falls back to CPU-only inference (0/43 layers on GPU).
unset CUDA_VISIBLE_DEVICES

echo "Starting ollama on $(hostname) (port 12434)..."
OLLAMA_HOST="localhost:12434" OLLAMA_MODELS="${OLLAMA_MODELS}" \
    "${OLLAMA_BIN}" serve &
OLLAMA_PID=$!

# Guarantee ollama is killed when the job exits (success or failure)
trap 'echo "Stopping ollama (PID ${OLLAMA_PID})..."; kill "${OLLAMA_PID}" 2>/dev/null; wait "${OLLAMA_PID}" 2>/dev/null' EXIT

# Wait up to 90 s for the server to accept connections
echo -n "Waiting for ollama to be ready"
for i in $(seq 1 90); do
    if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo " ready (${i}s)"
        break
    fi
    if [ "${i}" -eq 90 ]; then
        echo ""
        echo "ERROR: ollama did not respond within 90 s on ${OLLAMA_HOST}."
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ── Pre-flight — verify simulation models can load ────────────────────────────
export USER_MODEL="${USER_MODEL:-gemma4:e4b}"
export CHATBOT_MODEL="${CHATBOT_MODEL:-teta-sft-v2:latest}"

echo ""
echo "Pre-flight — verifying Ollama model '${USER_MODEL}' can load..."
WARMUP_TMPFILE=$(mktemp /tmp/ollama_warmup_XXXXXX.json)
WARMUP_HTTP=$(curl -s -o "${WARMUP_TMPFILE}" -w "%{http_code}" \
    -X POST "${OLLAMA_HOST}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${USER_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 3}" \
    --max-time 120)
if [ "${WARMUP_HTTP}" != "200" ]; then
    echo "ERROR: model '${USER_MODEL}' failed to load (HTTP ${WARMUP_HTTP})."; cat "${WARMUP_TMPFILE}"
    rm -f "${WARMUP_TMPFILE}"; exit 1
fi
rm -f "${WARMUP_TMPFILE}"
echo "  Model is ready."

# ── Output / log paths ────────────────────────────────────────────────────────
XLSX="${PROJECT_ROOT}/data/simulation/skenario_mental.xlsx"
OUTPUT_DIR="${PROJECT_ROOT}/results/PsyDial"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/baseline_psydial_${TIMESTAMP}.log"

# ── Run simulation — tee stdout+stderr to log ─────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output dir: ${OUTPUT_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Args: $*"

python3 "${SCRIPT_DIR}/simulate_psydial.py" \
    --xlsx        "${XLSX}" \
    --output-dir  "${OUTPUT_DIR}" \
    --ollama-url  "${OLLAMA_HOST}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished. Log saved to: ${LOG_FILE}"
