'use client'

import { motion } from 'framer-motion'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { useState, useEffect, useRef } from 'react'

export default function SimulationViewer() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!isPlaying) return

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          setIsPlaying(false)
          return 100
        }
        return prev + 1
      })
    }, 50)

    return () => clearInterval(interval)
  }, [isPlaying])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#1F2937'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw track
    ctx.strokeStyle = '#4B5563'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.arc(400, 300, 200, 0, Math.PI * 2)
    ctx.stroke()

    // Draw obstacles
    ctx.fillStyle = '#EF4444'
    ctx.beginPath()
    ctx.arc(300, 200, 30, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(500, 400, 30, 0, Math.PI * 2)
    ctx.fill()

    // Draw goal gate
    ctx.fillStyle = '#10B981'
    ctx.fillRect(580, 280, 40, 40)

    // Draw vehicle (animated based on progress)
    const angle = (progress / 100) * Math.PI * 2
    const x = 400 + Math.cos(angle) * 200
    const y = 300 + Math.sin(angle) * 200

    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle + Math.PI / 2)
    
    // Car body
    ctx.fillStyle = '#3B82F6'
    ctx.fillRect(-15, -25, 30, 50)
    
    // Drift effect
    if (isPlaying) {
      ctx.strokeStyle = '#60A5FA'
      ctx.lineWidth = 2
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(0, 25)
      ctx.lineTo(0, 40)
      ctx.stroke()
    }
    
    ctx.restore()

    // Draw trajectory path
    ctx.strokeStyle = '#8B5CF6'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.beginPath()
    for (let i = 0; i <= progress; i++) {
      const a = (i / 100) * Math.PI * 2
      const px = 400 + Math.cos(a) * 200
      const py = 300 + Math.sin(a) * 200
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()

  }, [progress, isPlaying])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="bg-gray-800/50 p-8 rounded-xl border border-gray-700"
    >
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-3xl font-bold">Drift Trajectory Simulation</h3>
        <div className="flex gap-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-all flex items-center gap-2"
          >
            {isPlaying ? (
              <>
                <Pause className="w-5 h-5" /> Pause
              </>
            ) : (
              <>
                <Play className="w-5 h-5" /> Play
              </>
            )}
          </button>
          <button
            onClick={() => {
              setProgress(0)
              setIsPlaying(false)
            }}
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-all flex items-center gap-2"
          >
            <RotateCcw className="w-5 h-5" /> Reset
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="w-full border border-gray-700 rounded-lg bg-gray-900"
      />

      <div className="mt-6">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Progress</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            style={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4 text-center">
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-gray-400 text-sm mb-1">Current Step</p>
          <p className="text-2xl font-bold text-blue-400">{Math.floor(progress / 2)}</p>
        </div>
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-gray-400 text-sm mb-1">Velocity</p>
          <p className="text-2xl font-bold text-purple-400">
            {isPlaying ? '2.5 m/s' : '0.0 m/s'}
          </p>
        </div>
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-gray-400 text-sm mb-1">Status</p>
          <p className="text-2xl font-bold text-green-400">
            {progress >= 100 ? 'Complete' : isPlaying ? 'Drifting' : 'Ready'}
          </p>
        </div>
      </div>
    </motion.div>
  )
}
