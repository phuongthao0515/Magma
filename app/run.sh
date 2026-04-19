#!/bin/bash
# UI Automation App — Run Commands
#
# Usage:
#   bash run.sh server                          # Start FastAPI backend (port 8000)
#   bash run.sh client                          # Start React frontend (port 4000)
#   bash run.sh agent                           # Start agent (heuristic stop mode)
#   bash run.sh agent --stop-mode terminate     # Start agent (terminate stop mode)
#   bash run.sh agent --stop-mode max_steps     # Start agent (max_steps only)
#   bash run.sh all                             # Start all 3 in background

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$1" in
    server)
        echo "Starting server on http://localhost:8000 ..."
        cd server && python -m src.main
        ;;
    client)
        echo "Starting client on http://localhost:4000 ..."
        cd client && npx vite --port 4000
        ;;
    agent)
        shift  # remove "agent" from args, pass the rest
        echo "Starting agent (args: $@) ..."
        python agent/executor.py --server-url http://localhost:8000 "$@"
        ;;
    all)
        echo "Starting all components..."
        echo "  Server: http://localhost:8000"
        echo "  Client: http://localhost:4000"
        echo "  Agent:  polling server"
        echo ""
        cd server && python -m src.main &
        SERVER_PID=$!
        cd client && npx vite --port 4000 &
        CLIENT_PID=$!
        sleep 10  # wait for server to load model
        python agent/executor.py --server-url http://localhost:8000 "${@:2}" &
        AGENT_PID=$!
        echo ""
        echo "PIDs: server=$SERVER_PID client=$CLIENT_PID agent=$AGENT_PID"
        echo "Press Ctrl+C to stop all"
        wait
        ;;
    *)
        echo "Usage: bash run.sh {server|client|agent|all}"
        echo ""
        echo "  server                          Start FastAPI backend (port 8000)"
        echo "  client                          Start React frontend (port 4000)"
        echo "  agent                           Start agent (server decides when to stop)"
        echo "  all                             Start all 3 components"
        ;;
esac
