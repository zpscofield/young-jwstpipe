#!/bin/bash
# Launch the YOUNG JWST pipeline UI locally. Streamlit opens your browser.
#
# Usage:
#   ./start.sh
#
# To share over a server, use ./share.sh instead.

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
    exit 1
fi

exec streamlit run app.py
