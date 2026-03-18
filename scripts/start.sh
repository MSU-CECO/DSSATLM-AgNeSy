#!/bin/bash

cd ~/DSSATLM-AgNeSy

LOGDIR=~/dssatlm-logs
mkdir -p $LOGDIR

echo "Starting Docker services..."
docker compose up -d

echo "Waiting for services to be healthy..."
sleep 5

# Kill any existing ngrok session
pkill ngrok 2>/dev/null
sleep 1

# Start ngrok inside a detached tmux session, logging to file
tmux new-session -d -s ngrok "ngrok http 8080 --log=stdout > $LOGDIR/ngrok.log 2>&1"

sleep 3

# Get public URL
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)

echo ""
echo "=== DSSATLM-AgNeSy is running ==="
docker compose ps
echo ""
echo "ngrok public URL: $NGROK_URL"
echo ""
echo "Logs:"
echo "  ngrok:       $LOGDIR/ngrok.log"
echo "  backend:     docker compose logs -f backend"
echo "  weather-api: docker compose logs -f weather-api"
echo "  nginx:       docker compose logs -f nginx"
echo ""
echo "To view ngrok session: tmux attach -t ngrok"
echo "To stop ngrok:         tmux kill-session -t ngrok"
echo "To stop everything:    docker compose down && tmux kill-session -t ngrok"
