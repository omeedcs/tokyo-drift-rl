'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import io, { Socket } from 'socket.io-client'

interface LiveSimulationStreamProps {
  isRunning: boolean
  modelName: string | null
}

export default function LiveSimulationStream({ isRunning, modelName }: LiveSimulationStreamProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const socketRef = useRef<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [fps, setFps] = useState(0)
  const [frameCount, setFrameCount] = useState(0)
  const lastFrameTimeRef = useRef(Date.now())

  useEffect(() => {
    // Connect to simulation backend
    const socket = io('http://localhost:5001', {
      transports: ['websocket'],
    })

    socket.on('connect', () => {
      console.log('Connected to simulation server')
      setIsConnected(true)
    })

    socket.on('disconnect', () => {
      console.log('Disconnected from simulation server')
      setIsConnected(false)
    })

    socket.on('simulation_frame', (data: { frame: string }) => {
      // Receive base64 encoded frame from pygame
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
        setFrameCount(prev => prev + 1)
      }
      img.src = `data:image/png;base64,${data.frame}`
    })

    socketRef.current = socket

    return () => {
      socket.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!socketRef.current) return

    if (isRunning && modelName) {
      // Start simulation
      socketRef.current.emit('start_simulation', {
        model: modelName,
        scenario: 'loose',
        max_steps: 500,
      })
    } else {
      // Stop simulation
      socketRef.current.emit('stop_simulation')
    }
  }, [isRunning, modelName])

  const handleRestart = () => {
    if (socketRef.current && modelName) {
      socketRef.current.emit('restart_simulation', {
        model: modelName,
        scenario: 'loose',
        max_steps: 500,
      })
      setFrameCount(0)
    }
  }

  return (
    <div className="relative w-full h-full bg-black">
      <canvas
        ref={canvasRef}
        className="w-full h-full object-contain"
        style={{ imageRendering: 'auto' }}
      />
      
      {/* Overlay when not running */}
      {!isRunning && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="text-center"
          >
            <div className="text-6xl mb-4">🎮</div>
            <p className="text-xl text-gray-400">Start simulation to view live feed</p>
          </motion.div>
        </div>
      )}

      {/* Connection status */}
      <div className="absolute top-4 left-4">
        <div className="flex items-center gap-2 px-3 py-2 bg-black/70 backdrop-blur-sm rounded-lg border border-cyber-primary/30">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-cyber-success animate-pulse' : 'bg-cyber-danger'}`} />
          <span className="text-xs font-mono text-white">
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
      </div>

      {/* Stats overlay */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-4 right-4 space-y-2"
        >
          <div className="px-3 py-2 bg-black/70 backdrop-blur-sm rounded-lg border border-cyber-accent/30 font-mono text-sm">
            <div className="text-cyber-accent">FPS: {fps}</div>
          </div>
          <div className="px-3 py-2 bg-black/70 backdrop-blur-sm rounded-lg border border-cyber-primary/30 font-mono text-sm">
            <div className="text-cyber-primary">Frames: {frameCount}</div>
          </div>
        </motion.div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleRestart}
          disabled={!isRunning}
          className="px-4 py-2 bg-cyber-secondary/20 hover:bg-cyber-secondary/40 border border-cyber-secondary rounded-lg text-sm font-mono transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ↻ RESTART
        </motion.button>
      </div>

      {/* Recording indicator */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute bottom-4 left-4"
        >
          <div className="flex items-center gap-2 px-3 py-2 bg-red-500/20 backdrop-blur-sm rounded-lg border border-red-500">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs font-mono text-red-500 font-bold">● REC</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
