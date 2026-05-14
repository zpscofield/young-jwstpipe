#!/bin/bash
# Launch the YOUNG JWST pipeline UI and expose it via a public HTTPS URL
# using a Cloudflare Tunnel. Use this when running on a server you SSH into.
# Share the printed URL with anyone -- they can open it in any browser.
#
# Usage:
#   ./share.sh
#
# The URL is fresh each session (random subdomain). Press Ctrl-C to stop.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PORT=8501
BIN_DIR="$SCRIPT_DIR/bin"
CLOUDFLARED_BIN="$BIN_DIR/cloudflared"
STREAMLIT_LOG="$SCRIPT_DIR/.streamlit/.streamlit.log"
TUNNEL_LOG="$SCRIPT_DIR/.streamlit/.tunnel.log"

mkdir -p "$BIN_DIR" "$(dirname "$STREAMLIT_LOG")"

# 1. Make sure streamlit is available.
if ! command -v streamlit &>/dev/null; then
    echo "Error: 'streamlit' not found on PATH."
    echo "Activate the pipeline environment first (conda activate young-jwstpipe)."
    exit 1
fi

# 2. Make sure cloudflared is available; download a local copy if not.
if command -v cloudflared &>/dev/null; then
    CLOUDFLARED=cloudflared
else
    if [ ! -x "$CLOUDFLARED_BIN" ]; then
        echo "[setup] cloudflared not found; downloading to ./bin/cloudflared ..."
        OS="$(uname -s)"
        ARCH="$(uname -m)"
        case "$OS-$ARCH" in
            Linux-x86_64)
                URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
                curl -fsSL -o "$CLOUDFLARED_BIN" "$URL"
                ;;
            Linux-aarch64|Linux-arm64)
                URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
                curl -fsSL -o "$CLOUDFLARED_BIN" "$URL"
                ;;
            Darwin-x86_64)
                URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
                curl -fsSL "$URL" | tar -xz -C "$BIN_DIR"
                ;;
            Darwin-arm64)
                URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
                curl -fsSL "$URL" | tar -xz -C "$BIN_DIR"
                ;;
            *)
                echo "Error: no cloudflared binary available for $OS-$ARCH."
                echo "Install it manually: https://developers.cloudflare.com/cloudflared/install"
                exit 1
                ;;
        esac
        chmod +x "$CLOUDFLARED_BIN"
        echo "[setup] cloudflared installed."
    fi
    CLOUDFLARED="$CLOUDFLARED_BIN"
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

# 4. Start streamlit in the background.
echo "[1/2] Starting Streamlit on port $PORT..."
streamlit run app.py \
    --server.headless true \
    --server.port "$PORT" \
    > "$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!

# Wait for it to bind.
for _ in $(seq 1 60); do
    if curl -fsS "http://localhost:$PORT" -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

# 5. Start the tunnel and grab the public URL from its log.
echo "[2/2] Opening Cloudflare Tunnel..."
: > "$TUNNEL_LOG"
"$CLOUDFLARED" tunnel --no-autoupdate --url "http://localhost:$PORT" \
    > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done

if [ -z "$URL" ]; then
    echo ""
    echo "[error] Could not get a tunnel URL. Cloudflared log:"
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
