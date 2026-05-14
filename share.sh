#!/bin/bash
# Launch the YOUNG JWST pipeline UI and expose it via a public HTTPS URL
# using localhost.run. Use this when running on a server you SSH into.
# Share the printed URL with anyone -- they can open it in any browser.
#
# Usage:
#   ./share.sh
#
# localhost.run uses a plain SSH reverse tunnel -- no binary to install,
# no account needed. The URL is random per session (e.g. abc123.lhr.life).
# Press Ctrl-C to stop the interface and close the tunnel.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PORT=8501
STREAMLIT_LOG="$SCRIPT_DIR/.streamlit/.streamlit.log"
TUNNEL_LOG="$SCRIPT_DIR/.streamlit/.tunnel.log"

mkdir -p "$(dirname "$STREAMLIT_LOG")"

# 1. Make sure streamlit is available.
if ! command -v streamlit &>/dev/null; then
    echo "Error: 'streamlit' not found on PATH."
    echo "Activate the pipeline environment first (conda activate young-jwstpipe)."
    exit 1
fi

# 2. Make sure ssh is available (used to open the localhost.run tunnel).
if ! command -v ssh &>/dev/null; then
    echo "Error: 'ssh' not found on PATH. Install OpenSSH client."
    exit 1
fi

# 3. Clean up child processes when the script exits.
STREAMLIT_PID=""
TUNNEL_PID=""
cleanup() {
    echo ""
    echo "[stop] Shutting down..."
    [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
    [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Spinner helper: run a loop printing a spinner + elapsed seconds until
# the supplied condition returns 0, or until the timeout is hit.
# Usage:
#   spin_until "Waiting for X" 60 "command that returns 0 when ready"
SPIN_FRAMES=('|' '/' '-' '\')
spin_until() {
    local label="$1"
    local timeout="$2"
    local condition="$3"
    local start now elapsed i=0 frame
    start=$(date +%s)
    while ! eval "$condition"; do
        now=$(date +%s)
        elapsed=$((now - start))
        if [ "$elapsed" -ge "$timeout" ]; then
            printf "\r\033[K"
            echo "  ✗ $label timed out after ${timeout}s."
            return 1
        fi
        frame=${SPIN_FRAMES[$((i % ${#SPIN_FRAMES[@]}))]}
        i=$((i + 1))
        printf "\r  %s %s (%ds elapsed)" "$frame" "$label" "$elapsed"
        sleep 0.15
    done
    printf "\r\033[K"
    echo "  ✓ $label"
    return 0
}

# 4. Start streamlit in the background.
echo "[1/2] Starting Streamlit on port $PORT..."
streamlit run app.py \
    --server.headless true \
    --server.port "$PORT" \
    > "$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!

spin_until "Waiting for Streamlit to bind" 60 \
    "curl -fsS 'http://localhost:$PORT' -o /dev/null 2>/dev/null" || {
        echo "[error] Streamlit did not start. Log tail:"
        tail -n 20 "$STREAMLIT_LOG"
        exit 1
    }

# 5. Open the localhost.run tunnel.
echo "[2/2] Opening localhost.run tunnel..."
: > "$TUNNEL_LOG"
ssh \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$SCRIPT_DIR/.streamlit/.lhr_known_hosts" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -R "80:localhost:$PORT" \
    nokey@localhost.run \
    > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for the URL to appear in the SSH output (or for ssh to give up).
spin_until "Connecting to localhost.run" 60 \
    "[ -n \"\$(grep -oE 'https://[a-z0-9-]+\\.(lhr\\.life|lhrtunnel\\.link)' \"$TUNNEL_LOG\" 2>/dev/null | head -1)\" ] || ! kill -0 $TUNNEL_PID 2>/dev/null" || true

URL=$(grep -oE 'https://[a-z0-9-]+\.(lhr\.life|lhrtunnel\.link)' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)

if [ -z "$URL" ]; then
    echo ""
    echo "[error] Could not get a tunnel URL. SSH output:"
    cat "$TUNNEL_LOG"
    exit 1
fi

# 6. Print the URL prominently.
echo ""
echo "==========================================================================="
echo ""
echo "  Open this URL in your browser (you can share it with anyone):"
echo ""
echo "    $URL"
echo ""
echo "==========================================================================="
echo ""
echo "Press Ctrl-C to stop the interface and close the tunnel."
echo ""

# Wait until one of the child processes exits.
wait
