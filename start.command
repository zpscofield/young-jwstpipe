#!/bin/bash
# macOS double-click launcher. Identical to start.sh but with the .command
# extension so Finder opens it in Terminal when double-clicked.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if ! command -v streamlit &>/dev/null; then
    echo "Error: 'streamlit' not found on PATH."
    echo ""
    echo "Activate the pipeline environment first, e.g.:"
    echo "  conda activate young-jwstpipe"
    echo "  # or"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

exec streamlit run app.py
