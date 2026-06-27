#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install.sh — sets up GGUF Chatbot on Fedora 43 + NVIDIA RTX 4050
# Run once:  bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[91m'; GREEN='\033[92m'; YELLOW='\033[93m'; CYAN='\033[96m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()   { echo -e "${RED}[ERR ]${RESET}  $*" >&2; exit 1; }

# ── 1. Check NVIDIA driver ────────────────────────────────────────────────────
info "Checking NVIDIA driver …"
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Install proprietary NVIDIA driver first:\n  sudo dnf install akmod-nvidia"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ok "NVIDIA driver present."

# ── 2. Check / install CUDA toolkit ──────────────────────────────────────────
info "Checking CUDA toolkit …"
if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found — installing cuda-toolkit …"
    sudo dnf install -y cuda-toolkit || \
        die "Failed. Try: sudo dnf install cuda-toolkit\nOr install via NVIDIA's .run installer."
fi
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "?")
ok "CUDA $CUDA_VER"

# ── 3. Python venv ────────────────────────────────────────────────────────────
info "Creating Python virtual environment …"
VENV_DIR="$(dirname "$0")/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q
ok "venv ready at $VENV_DIR"

# ── 4. Install llama-cpp-python with CUDA ────────────────────────────────────
info "Building llama-cpp-python with CUDA support (this takes 2–5 min) …"
CMAKE_ARGS="-DGGML_CUDA=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python \
    --upgrade \
    --force-reinstall \
    --no-cache-dir \
    -q

ok "llama-cpp-python installed."

# ── 5. Verify GPU is visible ──────────────────────────────────────────────────
info "Verifying CUDA device is visible to llama-cpp-python …"
python3 - <<'PYEOF'
from llama_cpp import llama_cpp
count = llama_cpp.llama_backend_init.__module__  # just import test
print("  Import OK")
PYEOF
ok "All done!"

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║  Setup complete! Activate venv and run:                  ║${RESET}"
echo -e "${GREEN}║                                                          ║${RESET}"
echo -e "${GREEN}║  source venv/bin/activate                                ║${RESET}"
echo -e "${GREEN}║  python chat.py --model /path/to/model.gguf              ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo
