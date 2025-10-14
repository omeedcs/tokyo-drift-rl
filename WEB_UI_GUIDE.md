# 🚀 Next.js Web UI - Complete Guide

## Overview

The **Next.js Web UI** replaces the Gradio interface with a **professional, production-ready web application** featuring:

### ✨ Key Features

1. **🎮 3D Environment Viewer**
   - Real-time Three.js 3D visualization
   - Vehicle with trail effects
   - Dynamic lighting and shadows
   - Interactive camera controls

2. **📺 Live Simulation Stream**
   - **Actual pygame simulation embedded** in the browser
   - WebSocket-based streaming (30 FPS)
   - No static plots - see the real simulation running!

3. **📈 Animated Trajectory**
   - Real-time path drawing with Canvas API
   - **Animated line that grows** as vehicle moves
   - Smooth gradient effects
   - No instant plots - watch it unfold!

4. **📊 Real-time Metrics**
   - Live performance data
   - Position, velocity, rewards
   - Updates every frame

5. **🎨 Premium Cyberpunk UI**
   - Dark theme with neon accents (cyan/magenta)
   - Smooth animations with Framer Motion
   - Glassmorphism effects
   - Responsive design

## 🚀 Quick Start

### One-Command Launch

```bash
./start_web_ui.sh
```

This script will:
1. Install dependencies (if needed)
2. Start backend server (port 5000)
3. Start frontend UI (port 3000)
4. Open `http://localhost:3000` in your browser

### Manual Launch

```bash
# Terminal 1: Backend
./venv/bin/pip install flask flask-socketio flask-cors
./venv/bin/python simulation_server.py

# Terminal 2: Frontend
cd web-ui
npm install
npm run dev
```

## 🎯 How It Works

### Architecture

```
┌─────────────────┐
│   Browser UI    │  ← Next.js (React)
│   localhost:3000│
└────────┬────────┘
         │ WebSocket
         ↓
┌─────────────────┐
│  Flask Server   │  ← Python
│   localhost:5000│
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ PyGame + Models │  ← Simulation
│   (SAC/IKD)     │
└─────────────────┘
```

### Data Flow

1. **User selects model** in browser
2. **Clicks START** → WebSocket message to server
3. **Server starts simulation** with selected model
4. **PyGame renders frames** → Captured as PNG
5. **Frames encoded to base64** → Sent via WebSocket
6. **Browser displays frames** in real-time
7. **3D view + metrics** update simultaneously

## 📁 Project Structure

```
autonomous-vehicle-drifting/
├── web-ui/                          # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Main page
│   │   │   ├── layout.tsx          # Root layout
│   │   │   └── globals.css         # Global styles
│   │   └── components/
│   │       ├── SimulationViewer3D.tsx       # 3D Three.js
│   │       ├── AnimatedTrajectory.tsx       # Canvas animation
│   │       ├── LiveSimulationStream.tsx     # WebSocket stream
│   │       ├── ModelSelector.tsx            # Model picker
│   │       ├── ControlPanel.tsx             # Start/Stop
│   │       └── MetricsPanel.tsx             # Stats display
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── simulation_server.py             # WebSocket backend
├── start_web_ui.sh                  # Launch script
└── WEB_UI_GUIDE.md                  # This file
```

## 🎮 Usage Guide

### Step 1: Select Model

Click on any model card to select it:
- **IKD Models**: Blue cards
- **SAC Models**: Purple cards

The selected model will show a checkmark.

### Step 2: Start Simulation

Click **"START SIMULATION"** button.

You'll see 4 panels update in real-time:

#### Panel 1: 3D Environment (Top-Left)
- Interactive 3D scene
- Vehicle with glowing trail
- Obstacles and goal gate
- Drag to rotate, scroll to zoom

#### Panel 2: Live Simulation (Top-Right)
- **Actual pygame window** embedded
- Shows the real simulation rendering
- FPS counter
- Frame count

#### Panel 3: Animated Trajectory (Bottom-Left)
- Canvas-based 2D view
- **Line animates as vehicle moves**
- Gradient trail effect
- Real-time path drawing

#### Panel 4: Metrics (Bottom-Right)
- Step count
- Total reward
- Velocity
- Angular velocity
- Position (X, Y)

### Step 3: Watch the Show!

- **3D vehicle** moves with trail effect
- **Live stream** shows actual simulation
- **Trajectory line** grows in real-time
- **Metrics** update every frame
- Everything is **synchronized**!

## 🎨 Customization

### Change Colors

Edit `web-ui/tailwind.config.ts`:

```typescript
colors: {
  cyber: {
    primary: '#00ffff',    // Cyan (change to any hex)
    secondary: '#ff00ff',  // Magenta
    accent: '#ffff00',     // Yellow
  },
}
```

### Adjust Frame Rate

Edit `simulation_server.py`:

```python
# Line 128
time.sleep(0.03)  # 30 FPS (lower = faster, higher = slower)
```

### Modify 3D Scene

Edit `web-ui/src/components/SimulationViewer3D.tsx`:
- Change vehicle colors
- Add more objects
- Modify lighting
- Adjust camera position

## 🔧 Technical Details

### Frontend Stack
- **Next.js 14**: React framework with app router
- **TypeScript**: Type-safe code
- **Three.js**: 3D graphics engine
- **@react-three/fiber**: React renderer for Three.js
- **@react-three/drei**: Three.js helpers
- **Framer Motion**: Animation library
- **Tailwind CSS**: Utility-first CSS
- **Socket.IO Client**: WebSocket communication

### Backend Stack
- **Flask**: Lightweight web framework
- **Flask-SocketIO**: WebSocket server
- **PyGame**: Simulation rendering
- **PIL**: Image processing
- **Base64**: Frame encoding

### Performance
- **Frontend**: 60 FPS (3D animation)
- **Backend**: 30 FPS (simulation streaming)
- **Latency**: <50ms (local network)
- **Resolution**: 800x800 (pygame default)

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### WebSocket Not Connecting

1. Check backend is running: `curl http://localhost:5000`
2. Check console for errors
3. Restart both servers
4. Clear browser cache

### 3D Scene Black Screen

1. Check WebGL support: Visit `https://get.webgl.org/`
2. Try different browser (Chrome/Firefox)
3. Update graphics drivers
4. Check console for Three.js errors

### Simulation Stream Not Showing

1. Verify pygame can render: `python -c "import pygame; pygame.init()"`
2. Check models exist in `trained_models/` or `dc_saves/`
3. Look at backend console for errors
4. Restart simulation server

## 📊 Comparison: Gradio vs Next.js

| Feature | Gradio | Next.js UI |
|---------|--------|------------|
| **3D Visualization** | ❌ None | ✅ Full 3D with Three.js |
| **Live Streaming** | ❌ Static images | ✅ Real-time WebSocket |
| **Animations** | ❌ Static plots | ✅ Smooth Canvas animations |
| **UI Quality** | ⚠️ Basic | ✅ Professional cyberpunk theme |
| **Customization** | ⚠️ Limited | ✅ Full control (React/Tailwind) |
| **Performance** | ⚠️ Slower updates | ✅ 60 FPS frontend, 30 FPS stream |
| **Embed Simulations** | ❌ No | ✅ Yes (real pygame) |
| **Production Ready** | ⚠️ Prototype | ✅ Yes |

## 🎯 Next Steps

### Enhancements You Can Add

1. **Multiple Camera Angles**: Add camera presets (top-down, side, chase)
2. **Record Videos**: Capture simulation as MP4
3. **Compare Models**: Run 2 models side-by-side
4. **Replay System**: Save and replay simulations
5. **3D Path Prediction**: Show predicted future path
6. **Model Upload**: Allow uploading new models via UI
7. **Dashboard**: Add statistics and leaderboards
8. **Mobile Support**: Optimize for tablets/phones

### Development Tips

```bash
# Run with type checking
cd web-ui
npm run lint

# Build for production
npm run build

# Run production build
npm start
```

## 📚 Documentation

- **Next.js**: https://nextjs.org/docs
- **Three.js**: https://threejs.org/docs
- **React Three Fiber**: https://docs.pmnd.rs/react-three-fiber
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Socket.IO**: https://socket.io/docs/v4/

## 🎉 Conclusion

This Next.js UI provides a **professional research platform** with:
- Real-time 3D visualization
- Embedded simulation streaming
- Smooth animations
- Production-ready code
- Beautiful cyberpunk aesthetic

**Launch it now:**
```bash
./start_web_ui.sh
```

Then open **`http://localhost:3000`** and enjoy! 🚀
