# 🏎️ Autonomous Drift Simulator - Next.js UI

**Premium real-time 3D visualization platform** with live simulation streaming, animated trajectories, and AI model comparison.

## ✨ Features

- 🎮 **3D Environment Viewer** - Real-time Three.js visualization with vehicle physics
- 📺 **Live Simulation Stream** - Stream actual pygame simulations via WebSocket
- 📈 **Animated Trajectory** - Real-time path drawing with Canvas API
- 📊 **Real-time Metrics** - Live performance tracking
- 🤖 **Model Selection** - Compare IKD and SAC models
- ⚡ **WebSocket Integration** - Low-latency simulation streaming
- 🎨 **Cyberpunk Theme** - Stunning dark UI with neon accents

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web-ui
npm install
```

### 2. Install Backend Dependencies

```bash
cd ..
pip install flask flask-socketio flask-cors python-socketio
```

### 3. Start Backend Server

```bash
# Terminal 1: Start simulation streaming server
python simulation_server.py
```

The backend will run on `http://localhost:5000`

### 4. Start Frontend

```bash
# Terminal 2: Start Next.js dev server
cd web-ui
npm run dev
```

The UI will open at `http://localhost:3000`

## 📁 Project Structure

```
web-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Main application page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   └── components/
│       ├── SimulationViewer3D.tsx      # 3D Three.js viewer
│       ├── AnimatedTrajectory.tsx      # Canvas-based animation
│       ├── LiveSimulationStream.tsx    # WebSocket video stream
│       ├── ModelSelector.tsx           # Model selection UI
│       ├── ControlPanel.tsx            # Simulation controls
│       └── MetricsPanel.tsx            # Real-time metrics
├── package.json
├── tailwind.config.ts
└── next.config.js

simulation_server.py              # Python WebSocket server
```

## 🎮 Usage

1. **Select a Model**: Choose from available IKD or SAC models
2. **Start Simulation**: Click "START SIMULATION" 
3. **Watch in Real-time**:
   - 🎮 3D view shows vehicle with trail effect
   - 📺 Live feed streams actual pygame simulation
   - 📈 Animated trajectory draws the path
   - 📊 Metrics update every frame

## 🔧 Technology Stack

### Frontend
- **Next.js 14** - React framework
- **Three.js** - 3D graphics
- **@react-three/fiber** - React Three.js renderer
- **@react-three/drei** - Three.js helpers
- **Framer Motion** - Animations
- **Tailwind CSS** - Styling
- **Socket.IO Client** - WebSocket client

### Backend
- **Flask** - Web framework
- **Flask-SocketIO** - WebSocket server
- **PyGame** - Simulation rendering
- **NumPy** - Data processing
- **PIL** - Image encoding

## 📡 WebSocket API

### Client → Server

- `start_simulation` - Start simulation with model/scenario
- `stop_simulation` - Stop current simulation
- `restart_simulation` - Restart with new parameters
- `get_available_models` - Fetch model list

### Server → Client

- `simulation_frame` - Video frame (base64 PNG)
- `simulation_metrics` - Real-time metrics
- `simulation_complete` - Episode finished
- `simulation_error` - Error occurred

## 🎨 Customization

### Change Theme Colors

Edit `tailwind.config.ts`:

```typescript
colors: {
  cyber: {
    primary: '#00ffff',    // Cyan
    secondary: '#ff00ff',  // Magenta
    accent: '#ffff00',     // Yellow
    // ... customize more
  },
}
```

### Adjust Simulation Settings

In `simulation_server.py`:

```python
# Frame rate
time.sleep(0.03)  # 30 FPS (adjust as needed)

# Resolution
env = GymDriftEnv(render_mode="rgb_array")  # Default 800x800
```

## 🐛 Troubleshooting

### WebSocket Connection Failed

```bash
# Check backend is running
curl http://localhost:5000

# Restart server
python simulation_server.py
```

### 3D Scene Not Rendering

- Check browser console for Three.js errors
- Ensure WebGL is enabled in browser
- Try different browser (Chrome/Firefox recommended)

### Simulation Lag

- Reduce frame rate in `simulation_server.py`
- Lower resolution in pygame renderer
- Check CPU usage

## 🚧 Development

```bash
# Install dev dependencies
npm install

# Run linter
npm run lint

# Build for production
npm run build

# Start production server
npm start
```

## 📝 Notes

- Backend requires trained models in `trained_models/` and `dc_saves/`
- First run will auto-discover available models
- WebSocket reconnects automatically on disconnect
- 3D scene updates at 60 FPS, simulation at 30 FPS

## 🤝 Contributing

This is a research platform. Feel free to:
- Add new visualization modes
- Improve 3D graphics
- Optimize streaming performance
- Add more metrics panels

## 📄 License

MIT License - See main repository

---

**Built with ❤️ for autonomous vehicle research**
