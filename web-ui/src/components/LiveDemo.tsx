'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import io, { Socket } from 'socket.io-client'

export default function LiveDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const socketRef = useRef<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedModel, setSelectedModel] = useState('sac_loose_2')
  const [modelType, setModelType] = useState<'ikd' | 'sac'>('sac')
  const [availableModels, setAvailableModels] = useState<{ikd: string[], sac: string[]}>({
    ikd: [],
    sac: ['sac_loose_2']
  })
  const [metrics, setMetrics] = useState({
    step: 0,
    reward: 0,
    totalReward: 0,
    velocity: 0,
    position: { x: 0, y: 0 },
  })
  const [fps, setFps] = useState(0)
  const [status, setStatus] = useState('Disconnected')
  const lastFrameTimeRef = useRef(Date.now())

  useEffect(() => {
    // Connect to simulation backend
    const socket = io('http://localhost:5001', {
      transports: ['websocket'],
    })

    socket.on('connect', () => {
      console.log('Connected to simulation server')
      setIsConnected(true)
      setStatus('Connected - Ready')
      
      // Request available models
      socket.emit('get_available_models')
    })

    socket.on('available_models', (data: { ikd: string[], sac: string[] }) => {
      console.log('Received models:', data)
      setAvailableModels(data)
      // Set default model based on what's available
      if (data.sac.length > 0) {
        setSelectedModel(data.sac[0])
        setModelType('sac')
      } else if (data.ikd.length > 0) {
        setSelectedModel(data.ikd[0])
        setModelType('ikd')
      }
    })

    socket.on('disconnect', () => {
      console.log('Disconnected from simulation server')
      setIsConnected(false)
      setStatus('Disconnected')
    })

    socket.on('simulation_frame', (data: { frame: string }) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const img = new Image()
      img.onload = () => {
        canvas.width = img.width
        canvas.height = img.height
        ctx.drawImage(img, 0, 0)

        // Update FPS
        const now = Date.now()
        const deltaTime = now - lastFrameTimeRef.current
        if (deltaTime > 0) {
          setFps(Math.round(1000 / deltaTime))
        }
        lastFrameTimeRef.current = now
      }
      img.src = `data:image/png;base64,${data.frame}`
    })

    socket.on('simulation_metrics', (data: any) => {
      setMetrics(prev => ({
        step: data.step || 0,
        reward: data.reward || 0,
        totalReward: prev.totalReward + (data.reward || 0),
        velocity: data.velocity || 0,
        position: data.position || { x: 0, y: 0 },
      }))
    })

    socket.on('simulation_started', () => {
      setIsRunning(true)
      setStatus('Running')
      setMetrics({
        step: 0,
        reward: 0,
        totalReward: 0,
        velocity: 0,
        position: { x: 0, y: 0 },
      })
    })

    socket.on('simulation_stopped', () => {
      setIsRunning(false)
      setStatus('Stopped')
    })

    socket.on('simulation_complete', (data: any) => {
      setIsRunning(false)
      setStatus(`Complete - ${data.success ? 'Success!' : 'Failed'}`)
    })

    socket.on('simulation_error', (data: any) => {
      setIsRunning(false)
      setStatus(`Error: ${data.error}`)
      console.error('Simulation error:', data.error)
    })

    socketRef.current = socket

    return () => {
      socket.disconnect()
    }
  }, [])

  const handleStart = () => {
    if (socketRef.current && !isRunning) {
      socketRef.current.emit('start_simulation', {
        model: selectedModel,
        scenario: 'loose',
        max_steps: 500,
      })
    }
  }

  const handleStop = () => {
    if (socketRef.current && isRunning) {
      socketRef.current.emit('stop_simulation')
    }
  }

  const handleRestart = () => {
    if (socketRef.current) {
      socketRef.current.emit('restart_simulation', {
        model: selectedModel,
        scenario: 'loose',
        max_steps: 500,
      })
    }
  }

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <div className="card">
        <h3 className="text-lg font-bold mb-4">Select Model & Type</h3>
        
        {/* Model Type Tabs */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => {
              setModelType('sac')
              if (availableModels.sac.length > 0) {
                setSelectedModel(availableModels.sac[0])
              }
            }}
            disabled={isRunning || availableModels.sac.length === 0}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-bold transition ${
              modelType === 'sac'
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            } ${(isRunning || availableModels.sac.length === 0) ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            SAC Models ({availableModels.sac.length})
          </button>
          <button
            onClick={() => {
              setModelType('ikd')
              if (availableModels.ikd.length > 0) {
                setSelectedModel(availableModels.ikd[0])
              }
            }}
            disabled={isRunning || availableModels.ikd.length === 0}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-bold transition ${
              modelType === 'ikd'
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            } ${(isRunning || availableModels.ikd.length === 0) ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            IKD Models ({availableModels.ikd.length})
          </button>
        </div>

        {/* Model List */}
        <div className="space-y-2">
          {(modelType === 'sac' ? availableModels.sac : availableModels.ikd).length > 0 ? (
            (modelType === 'sac' ? availableModels.sac : availableModels.ikd).map((model) => (
              <button
                key={model}
                className={`w-full px-4 py-3 text-sm border-2 rounded font-mono font-bold transition text-left ${
                  selectedModel === model
                    ? 'border-black bg-gray-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
                onClick={() => setSelectedModel(model)}
                disabled={isRunning}
              >
                <div className="flex items-center justify-between">
                  <span>{model}</span>
                  {selectedModel === model && <span className="text-green-600">✓</span>}
                </div>
              </button>
            ))
          ) : (
            <div className="text-center py-6 text-gray-500 text-sm">
              No {modelType.toUpperCase()} models available. Train some models first.
            </div>
          )}
        </div>

        <div className="mt-3 p-3 bg-gray-50 rounded text-xs text-gray-600">
          <strong>Selected:</strong> <span className="font-mono">{selectedModel}</span> ({modelType.toUpperCase()})
        </div>
      </div>

      {/* Connection Status */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 rounded border border-gray-200">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-sm font-medium">{status}</span>
        </div>
        {isRunning && (
          <div className="flex items-center gap-2 text-sm">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="font-mono font-bold">REC</span>
          </div>
        )}
      </div>

      {/* Simulation Viewer */}
      <div className="bg-black rounded-lg overflow-hidden border-2 border-gray-200 relative" style={{ aspectRatio: '16/9' }}>
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
          style={{ imageRendering: 'auto' }}
        />
        
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <div className="text-center text-white">
              <div className="text-6xl mb-4">🏎️</div>
              <p className="text-lg mb-2">Simulation Stream</p>
              <p className="text-sm text-gray-400">
                {isConnected ? 'Click Start to begin' : 'Connecting to server...'}
              </p>
            </div>
          </div>
        )}

        {/* FPS Counter */}
        {isRunning && (
          <div className="absolute top-4 right-4 px-3 py-1 bg-black/70 rounded text-white text-sm font-mono">
            {fps} FPS
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-4">
        <button 
          onClick={handleStart}
          disabled={!isConnected || isRunning}
          className={`btn-primary flex-1 ${(!isConnected || isRunning) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          ▶ Start Simulation
        </button>
        <button 
          onClick={handleStop}
          disabled={!isRunning}
          className={`btn-secondary ${!isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          ⏸ Stop
        </button>
        <button 
          onClick={handleRestart}
          disabled={!isConnected}
          className={`btn-secondary ${!isConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          ↻ Restart
        </button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div 
          className="card text-center"
          animate={{ scale: isRunning ? [1, 1.02, 1] : 1 }}
          transition={{ duration: 0.3 }}
        >
          <div className="text-sm text-gray-600 mb-1">Step</div>
          <div className="text-2xl font-bold font-mono">{metrics.step}</div>
        </motion.div>
        
        <motion.div 
          className="card text-center"
          animate={{ scale: metrics.reward !== 0 ? [1, 1.05, 1] : 1 }}
          transition={{ duration: 0.2 }}
        >
          <div className="text-sm text-gray-600 mb-1">Reward</div>
          <div className="text-2xl font-bold font-mono">{metrics.reward.toFixed(2)}</div>
        </motion.div>
        
        <div className="card text-center">
          <div className="text-sm text-gray-600 mb-1">Velocity</div>
          <div className="text-2xl font-bold font-mono">{metrics.velocity.toFixed(1)} <span className="text-sm text-gray-500">m/s</span></div>
        </div>
        
        <div className="card text-center">
          <div className="text-sm text-gray-600 mb-1">Total Reward</div>
          <div className="text-2xl font-bold font-mono">{metrics.totalReward.toFixed(1)}</div>
        </div>
      </div>

      {/* Position Display */}
      {isRunning && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex justify-between items-center">
            <div>
              <div className="text-xs text-gray-600 mb-1">Position</div>
              <div className="font-mono text-sm">
                X: <span className="font-bold">{metrics.position.x.toFixed(2)}</span> m, 
                Y: <span className="font-bold ml-2">{metrics.position.y.toFixed(2)}</span> m
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-600 mb-1">Frame Rate</div>
              <div className="font-mono text-sm font-bold">{fps} FPS</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      <div className="p-6 bg-gray-50 rounded-lg border border-gray-200">
        <h4 className="font-bold mb-3">How to Use</h4>
        <ol className="space-y-2 text-sm text-gray-700">
          <li><strong>1.</strong> Wait for "Connected - Ready" status (green dot)</li>
          <li><strong>2.</strong> Click <strong>"Start Simulation"</strong> to begin</li>
          <li><strong>3.</strong> Watch the agent navigate in real-time with live metrics</li>
          <li><strong>4.</strong> Click <strong>"Stop"</strong> to pause or <strong>"Restart"</strong> to reset</li>
          <li><strong>5.</strong> Metrics update automatically as the simulation runs</li>
        </ol>
        <div className="mt-4 p-3 bg-white rounded border border-gray-200">
          <p className="text-xs text-gray-600 mb-1"><strong>Note:</strong></p>
          <p className="text-xs text-gray-600">
            This demo connects to the WebSocket server on <code className="bg-gray-100 px-1 rounded">localhost:5001</code>.
            The server must be running for the demo to work. You should see "Connected - Ready" above.
          </p>
        </div>
      </div>
    </div>
  )
}
