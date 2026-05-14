#!/bin/bash
# Launch the YOUNG JWST pipeline UI.
#
# Local use: just run ./start.sh (or double-click start.command on macOS).
# Streamlit opens your browser automatically.
#
# Remote use (you're SSH'd into a server): the script detects the SSH
# session, prints the exact 'ssh -L ...' command you need to run on your
# laptop, and starts Streamlit in headless mode so it doesn't try to
# open a browser on the server.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PORT=8501

if ! command -v streamlit &>/dev/null; then
    echo "Error: 'streamlit' not found on PATH."
    echo ""
    echo "Activate the pipeline environment first, e.g.:"
    echo "  conda activate young-jwstpipe"
    echo "  # or"
    echo "  pip install -r requirements.txt"
    exit 1
fi

if [ -n "${SSH_CONNECTION:-}" ]; then
    # We're inside an SSH session on a server. Print port-forward instructions.
    # SSH_CONNECTION = "<client_ip> <client_port> <server_ip> <server_port>"
    SSH_PORT=$(echo "$SSH_CONNECTION" | awk '{print $4}')
    SERVER_HOST=$(hostname -f 2>/dev/null || hostname)
    WHOAMI=${USER:-$(whoami)}

    if [ "$SSH_PORT" = "22" ] || [ -z "$SSH_PORT" ]; then
        SSH_CMD="ssh -L $PORT:localhost:$PORT $WHOAMI@$SERVER_HOST"
    else
        SSH_CMD="ssh -p $SSH_PORT -L $PORT:localhost:$PORT $WHOAMI@$SERVER_HOST"
    fi

    echo "==========================================================================="
    echo ""
    echo "  You're running on a server over SSH. The pipeline UI starts below."
    echo ""
    echo "  If you're using VSCode or Cursor with Remote-SSH, the editor will"
    echo "  usually auto-forward port $PORT and offer a clickable 'Open in"
    echo "  Browser' popup -- in that case you can ignore the next step."
    echo ""
    echo "  Otherwise, you may need to forward the port yourself. On your"
    echo "  laptop, in a new terminal, run:"
    echo ""
    echo "       $SSH_CMD"
    echo ""
    echo "     (If you have a host alias in your ~/.ssh/config you can use that"
    echo "      instead: 'ssh -L $PORT:localhost:$PORT <your-alias>'.)"
    echo ""
    echo "  Then open http://localhost:$PORT in your laptop's browser."
    echo ""
    echo "  Press Ctrl-C here to stop the pipeline server."
    echo ""
    echo "==========================================================================="
    echo ""

    exec streamlit run app.py --server.headless true --server.port "$PORT"
else
    # Local. Streamlit will auto-open the browser.
    exec streamlit run app.py --server.port "$PORT"
fi
