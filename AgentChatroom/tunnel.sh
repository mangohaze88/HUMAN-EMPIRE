#!/bin/bash
# Start the Agent Chatroom server and expose it via Cloudflare Tunnel.
# Usage: bash tunnel.sh
#
# This gives you a temporary public URL (e.g., https://xyz.trycloudflare.com)
# that anyone can use to connect to your chatroom.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Agent Chatroom server..."
python main.py --web --no-constitution &
SERVER_PID=$!

# Wait for server to be ready
sleep 3

echo ""
echo "Starting Cloudflare Tunnel..."
echo "Your public URL will appear below:"
echo ""

cloudflared tunnel --url http://localhost:9000

# Clean up server when tunnel exits
kill $SERVER_PID 2>/dev/null
