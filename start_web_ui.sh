#!/bin/bash

echo "=========================================="
echo "🏎️  Autonomous Drift Simulator"
echo "=========================================="
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check if python dependencies are installed
echo "📦 Checking Python dependencies..."
./venv/bin/python -c "import flask" 2>/dev/null || {
    echo "📥 Installing Flask dependencies..."
    ./venv/bin/pip install flask flask-socketio flask-cors python-socketio
}

# Install npm dependencies if needed
if [ ! -d "web-ui/node_modules" ]; then
    echo "📥 Installing npm dependencies..."
    cd web-ui && npm install && cd ..
fi

echo ""
echo "✅ All dependencies ready!"
echo ""
echo "=========================================="
echo "Starting servers..."
echo "=========================================="
echo ""

# Start Python backend in background
echo "🚀 Starting simulation server on port 5001..."
./venv/bin/python simulation_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start Next.js frontend
echo "🌐 Starting web UI on port 3000..."
cd web-ui
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "✅ Servers running!"
echo "=========================================="
echo ""
echo "🔗 Open in browser: http://localhost:3000"
echo ""
echo "📡 Backend:  http://localhost:5001"
echo "🌐 Frontend: http://localhost:3000"
echo ""
echo "⚠️  Note: Port 5000 is used by AirPlay on macOS, using 5001 instead"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
