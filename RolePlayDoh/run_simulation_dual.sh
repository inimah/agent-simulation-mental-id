#!/bin/bash
# ── Conda ──────────────────────────────────────────────────────────────────────
source CONDA_HOME
conda activate env-name
# run_simulation_dual.sh — Baseline D2: Roleplay-doh Dual Adherence with AI-aware agents
# ------------------------------------------------------------------
# Launches baseline_simulator_4/simulate_conversation_dual.py which applies
# principle-adherence prompting to BOTH agents:
#
#   USER AGENT    — 8 Roleplay-doh patient principles (agent-type + category)
#   CHATBOT AGENT — 6 Indonesian adolescent counseling principles (CBT,
#                   person-centered, trauma-informed, cultural empathy),
#                   injected into the system prompt at each turn
#
# Usage:
#   sbatch baseline_simulator_4/run_simulation_dual.sh
#   bash   baseline_simulator_4/run_simulation_dual.sh --dry-run
#   bash   baseline_simulator_4/run_simulation_dual.sh --scenario-ids 1 5 12
#   bash   baseline_simulator_4/run_simulation_dual.sh --agent-type pendiam
#   bash   baseline_simulator_4/run_simulation_dual.sh --resume
#
# Overrides (set before calling):
#   USER_MODEL=gemma4:e4b            (default)
#   CHATBOT_MODEL=teta-sft-v2:latest (default)
#   MANIFEST_DIR=data/simulated-sft-fin  (default — set to "" to disable)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-PROJECT_ROOT/RolePlayDoh}"


# ── Ollama server ─────────────────────────────────────────────────────────────
export OLLAMA_HOST="http://localhost:12434"
export OLLAMA_MODELS="$HOME/.ollama/models"
export PATH="$HOME/.local/ollama/bin:$PATH"
OLLAMA_BIN="$HOME/.local/ollama/bin/ollama"

unset CUDA_VISIBLE_DEVICES

echo "Starting ollama on $(hostname) (port 12434)..."
OLLAMA_HOST="localhost:12434" OLLAMA_MODELS="${OLLAMA_MODELS}" \
    "${OLLAMA_BIN}" serve &
OLLAMA_PID=$!

trap 'echo "Stopping ollama (PID ${OLLAMA_PID})..."; kill "${OLLAMA_PID}" 2>/dev/null; wait "${OLLAMA_PID}" 2>/dev/null' EXIT

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

# ── Model configuration ────────────────────────────────────────────────────────
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


export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# ── Output / log paths ────────────────────────────────────────────────────────
XLSX="${PROJECT_ROOT}/data/simulation/skenario_mental.xlsx"
OUTPUT_DIR="${PROJECT_ROOT}/results/RolePlayDoh-aiaware"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/data/simulated-sft_baseline}"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/simulation_roleplaydoh_dual_${TIMESTAMP}.log"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Baseline — Roleplay-doh Dual Adherence Simulation (AI-aware)"
echo "══════════════════════════════════════════════════════════════"
echo "  User model    : ${USER_MODEL}"
echo "  Chatbot model : ${CHATBOT_MODEL}"
echo "  Manifest dir  : ${MANIFEST_DIR:-<none — random sampling>}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "  Log           : ${LOG_FILE}"
echo "  Extra args    : $*"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Run simulation ────────────────────────────────────────────────────────────
MANIFEST_FLAG=""
if [[ -n "${MANIFEST_DIR}" && -d "${MANIFEST_DIR}" ]]; then
    MANIFEST_FLAG="--manifest ${MANIFEST_DIR}"
elif [[ -n "${MANIFEST_DIR}" ]]; then
    echo "WARN: MANIFEST_DIR='${MANIFEST_DIR}' is not a directory — running without manifest (random sampling)."
fi

python3 "${SCRIPT_DIR}/simulate_conversation_dual.py" \
    --xlsx        "${XLSX}"       \
    --output-dir  "${OUTPUT_DIR}" \
    --ollama-url  "${OLLAMA_HOST}" \
    ${MANIFEST_FLAG}              \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished. Log saved to: ${LOG_FILE}"
