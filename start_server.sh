#!/bin/bash

# RAG API Server Startup Script
# This script starts the FastAPI server and optionally sets up Tailscale Funnel

echo "🚀 Starting RAG API Server..."
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the pyapi directory."
    exit 1
fi

# Default values
HOST="0.0.0.0"
PORT=8000
RELOAD="--reload"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-reload)
            RELOAD=""
            shift
            ;;
        --funnel)
            SETUP_FUNNEL=true
            shift
            ;;
        --help)
            echo "Usage: ./start_server.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT       Port to run the server on (default: 8000)"
            echo "  --no-reload       Disable auto-reload on code changes"
            echo "  --funnel          Set up Tailscale Funnel after starting server"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./start_server.sh                    # Start with defaults"
            echo "  ./start_server.sh --port 8080        # Start on port 8080"
            echo "  ./start_server.sh --funnel           # Start and setup Tailscale Funnel"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "❌ Error: uvicorn not found. Please install it:"
    echo "   pip install uvicorn"
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Auto-reload: $([ -n "$RELOAD" ] && echo "enabled" || echo "disabled")"
echo ""

# Start the server
# Start the server
echo "Starting FastAPI server in background..."
echo "API will be available at: http://localhost:$PORT"
echo "API Documentation: http://localhost:$PORT/docs"
echo ""

# Define log and pid files
LOG_FILE="server.log"
PID_FILE="server.pid"

# If funnel is requested, show instructions
if [ "$SETUP_FUNNEL" = true ]; then
    echo "📡 Tailscale Funnel instructions:"
    echo "   Run: tailscale funnel $PORT"
    echo ""
fi

# Start uvicorn in background
nohup uvicorn main:app --host $HOST --port $PORT $RELOAD > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID
echo $SERVER_PID > "$PID_FILE"

echo "✅ Server started with PID: $SERVER_PID"
echo "📄 Output redirected to: $LOG_FILE"
echo "🛑 To stop the server, run: kill \$(cat $PID_FILE)"
