#!/bin/bash

# Master script to fix the chaos and run everything

echo "======================================================================"
echo "🔧 FIXING THE CHAOS - Autonomous Vehicle Drifting Project"
echo "======================================================================"
echo ""

show_menu() {
    echo "What do you want to do?"
    echo ""
    echo "  1) 🎓 Train new SAC model (Research-Grade)"
    echo "  2) 🎮 Train new SAC model (Toy Mode - Faster)"
    echo "  3) 🧪 Quick test training (5k steps)"
    echo "  4) 🌐 Start Web UI with NEW server"
    echo "  5) 🌐 Start Web UI with OLD server"
    echo "  6) 📊 Test advanced features"
    echo "  7) 📋 List available models"
    echo "  8) 🧹 Clean up (remove pycache, logs)"
    echo "  9) ❌ Exit"
    echo ""
}

train_sac_research() {
    echo ""
    echo "======================================================================"
    echo "🎓 Training SAC with Research-Grade Features"
    echo "======================================================================"
    echo ""
    echo "This will train SAC on the advanced drift_gym with:"
    echo "  ✅ Sensor noise (GPS drift, IMU bias)"
    echo "  ✅ Perception pipeline (false positives/negatives)"
    echo "  ✅ Latency modeling (100ms delay)"
    echo "  ✅ 3D dynamics (weight transfer)"
    echo "  ✅ Moving obstacles"
    echo ""
    echo "Training time: ~20-30 minutes on M1 Max"
    echo ""
    read -p "Continue? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        source venv/bin/activate
        echo "1" | python train_sac_advanced.py
    fi
}

train_sac_toy() {
    echo ""
    echo "======================================================================"
    echo "🎮 Training SAC with Toy Mode (No Advanced Features)"
    echo "======================================================================"
    echo ""
    echo "Faster training, good for debugging and initial testing."
    echo "Training time: ~10-15 minutes on M1 Max"
    echo ""
    read -p "Continue? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        source venv/bin/activate
        echo "2" | python train_sac_advanced.py
    fi
}

quick_test() {
    echo ""
    echo "======================================================================"
    echo "🧪 Quick Test Training (5k steps)"
    echo "======================================================================"
    echo ""
    echo "Fast validation run (~3-5 minutes)"
    echo ""
    source venv/bin/activate
    echo "3" | python train_sac_advanced.py
}

start_web_ui_new() {
    echo ""
    echo "======================================================================"
    echo "🌐 Starting Web UI with NEW Server"
    echo "======================================================================"
    echo ""
    echo "Using simulation_server_v2.py (supports advanced models)"
    echo ""
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        echo "❌ npm is not installed. Please install Node.js first."
        echo "   Download from: https://nodejs.org/"
        return
    fi
    
    # Install Python dependencies
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
    echo "======================================================================"
    echo "Starting servers..."
    echo "======================================================================"
    echo ""
    
    # Start Python backend in background
    echo "🚀 Starting NEW simulation server on port 5001..."
    ./venv/bin/python simulation_server_v2.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 3
    
    # Start Next.js frontend
    echo "🌐 Starting web UI on port 3000..."
    cd web-ui
    npm run dev &
    FRONTEND_PID=$!
    
    echo ""
    echo "======================================================================"
    echo "✅ Servers running!"
    echo "======================================================================"
    echo ""
    echo "🔗 Open in browser: http://localhost:3000"
    echo ""
    echo "📡 Backend:  http://localhost:5001"
    echo "🌐 Frontend: http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop all servers"
    echo ""
    
    # Wait for Ctrl+C
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
    wait
}

start_web_ui_old() {
    echo ""
    echo "======================================================================"
    echo "🌐 Starting Web UI with OLD Server"
    echo "======================================================================"
    echo ""
    echo "Using simulation_server.py (original)"
    echo ""
    ./start_web_ui.sh
}

test_advanced_features() {
    echo ""
    echo "======================================================================"
    echo "📊 Testing Advanced Features"
    echo "======================================================================"
    echo ""
    source venv/bin/activate
    python drift_gym/examples/test_advanced_features.py
}

list_models() {
    echo ""
    echo "======================================================================"
    echo "📋 Available Models"
    echo "======================================================================"
    echo ""
    
    echo "SAC Models:"
    echo "----------"
    if [ -d "sac_advanced_models" ]; then
        ls -1 sac_advanced_models/ 2>/dev/null || echo "  (none)"
    else
        echo "  (no sac_advanced_models directory)"
    fi
    
    if [ -d "dc_saves" ]; then
        echo ""
        echo "Old SAC Models (dc_saves):"
        ls -1 dc_saves/ | grep -E "sac_" 2>/dev/null || echo "  (none)"
    fi
    
    echo ""
    echo "IKD Models:"
    echo "----------"
    if [ -d "trained_models" ]; then
        ls -1 trained_models/*.pt 2>/dev/null | xargs -n 1 basename || echo "  (none)"
    else
        echo "  (no trained_models directory)"
    fi
    
    echo ""
}

cleanup() {
    echo ""
    echo "======================================================================"
    echo "🧹 Cleaning Up"
    echo "======================================================================"
    echo ""
    
    echo "Removing __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "✅ Removed __pycache__"
    
    echo "Removing .pyc files..."
    find . -type f -name "*.pyc" -delete 2>/dev/null
    echo "✅ Removed .pyc files"
    
    if [ -f "sac_training.log" ]; then
        echo "Removing training log..."
        rm sac_training.log
        echo "✅ Removed sac_training.log"
    fi
    
    echo ""
    echo "✅ Cleanup complete!"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice (1-9): " choice
    
    case $choice in
        1)
            train_sac_research
            ;;
        2)
            train_sac_toy
            ;;
        3)
            quick_test
            ;;
        4)
            start_web_ui_new
            ;;
        5)
            start_web_ui_old
            ;;
        6)
            test_advanced_features
            ;;
        7)
            list_models
            ;;
        8)
            cleanup
            ;;
        9)
            echo ""
            echo "👋 Goodbye!"
            echo ""
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Invalid choice. Please enter 1-9."
            echo ""
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
