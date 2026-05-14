#!/bin/bash
# macOS double-click launcher. Forwards to start.sh, which decides
# between the local (open browser) and SSH (print port-forward instructions)
# paths automatically.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/start.sh" || {
    echo ""
    read -p "Press Enter to close..."
}
