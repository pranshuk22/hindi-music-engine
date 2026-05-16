#!/usr/bin/env bash

# ==============================================================================
# 🎧 Hindi Music Similarity Engine - Master Pipeline Runner
# ==============================================================================
# Strict mode: Exit on undeclared variables, but allow command failures to be handled
set -u

# --- Terminal Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper Functions ---
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Argument Parsing ---
show_help() {
    echo -e "${CYAN}Usage:${NC} ./run_all.sh [OPTIONS]"
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  -c, --csv FILE      Path to the source CSV file (default: data/songs.csv)"
    echo "  -i, --index NUM     Index to resume from (default: 0)"
    echo "  -h, --help          Show this help message and exit"
    echo ""
    echo "Example: ./run_all.sh --csv data/my_playlist.csv --index 15"
}

CSV_FILE="data/songs.csv"
START_INDEX=0
WORKERS=2

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--csv) CSV_FILE="$2"; shift ;;
        -i|--index) START_INDEX="$2"; shift ;;
        -w|--workers) WORKERS="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) warn "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# --- Environment Setup ---
mkdir -p logs data index

# Create a timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

clear
cat << "EOF"
================================================================
     🎧 HINDI MUSIC SIMILARITY ENGINE - PIPELINE RUNNER
================================================================
EOF

info "Initializing Pipeline..."
info "Source Data : $CSV_FILE"
info "Start Index : $START_INDEX"
info "Log File    : $LOG_FILE"
echo "----------------------------------------------------------------"

# --- Activate Virtual Environment ---
if [ -d "venv" ]; then
    info "Activating local virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    info "Activating local virtual environment (.venv)..."
    source .venv/bin/activate
else
    warn "No virtual environment found. Running with global python."
fi

echo "----------------------------------------------------------------"
info "Phase 1: Ingestion & Extraction"
info "Steps: Download ➔ Trim ➔ Demucs Stem ➔ Librosa Math ➔ CLAP ➔ NLP ➔ Clean"
echo "----------------------------------------------------------------"

# Run the python script, sending errors to the timestamped log
python scripts/run_pipeline.py "$START_INDEX" 2>> "$LOG_FILE"

if [ $? -ne 0 ]; then
    warn "Phase 1 finished with warnings/errors. Review: $LOG_FILE"
else
    success "Phase 1 completed successfully."
fi

echo "----------------------------------------------------------------"
info "Phase 2: Hybrid FAISS Indexing"
info "Steps: L2 Normalization ➔ Feature Scaling ➔ Vector Fusion ➔ Index Generation"
echo "----------------------------------------------------------------"

python index/build_index.py 2>> "$LOG_FILE"

if [ $? -ne 0 ]; then
    error "Phase 2 failed! Could not build the FAISS index."
    info "Check the logs: $LOG_FILE"
    exit 1
else
    success "Phase 2 completed. Vector index is ready."
fi

echo "================================================================"
success "🎉 Pipeline Execution Finished Successfully!"
echo -e "Launch the user interface by running: ${CYAN}streamlit run app.py${NC}"
echo "================================================================"