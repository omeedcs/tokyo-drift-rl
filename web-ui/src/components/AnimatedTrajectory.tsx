'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

interface Point {
  x: number
  y: number
  timestamp: number
}

interface AnimatedTrajectoryProps {
  isRunning: boolean
  onDataUpdate: (data: any) => void
}

export default function AnimatedTrajectory({ isRunning, onDataUpdate }: AnimatedTrajectoryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [points, setPoints] = useState<Point[]>([])
  const [currentPoint, setCurrentPoint] = useState<Point | null>(null)
  const animationFrameRef = useRef<number>()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Clear canvas
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)'
    ctx.lineWidth = 1
    const gridSize = 50
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Draw goal gate
    const gateX = canvas.width * 0.8
    const gateY = canvas.height / 2
    const gateWidth = 100
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 6
    ctx.beginPath()
    ctx.moveTo(gateX, gateY - gateWidth / 2)
    ctx.lineTo(gateX, gateY + gateWidth / 2)
    ctx.stroke()

    // Draw obstacles
    const obstacles = [
      { x: canvas.width * 0.4, y: canvas.height * 0.6, radius: 30 },
      { x: canvas.width * 0.5, y: canvas.height * 0.3, radius: 25 },
      { x: canvas.width * 0.6, y: canvas.height * 0.7, radius: 35 },
    ]
    obstacles.forEach(obs => {
      ctx.fillStyle = 'rgba(255, 0, 0, 0.3)'
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    })

    // Draw trajectory with gradient
    if (points.length > 1) {
      const gradient = ctx.createLinearGradient(
        points[0].x,
        points[0].y,
        points[points.length - 1].x,
        points[points.length - 1].y
      )
      gradient.addColorStop(0, '#00ffff')
      gradient.addColorStop(0.5, '#ff00ff')
      gradient.addColorStop(1, '#ffff00')

      ctx.strokeStyle = gradient
      ctx.lineWidth = 4
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.shadowBlur = 15
      ctx.shadowColor = '#00ffff'

      // Draw animated line segments
      ctx.beginPath()
      ctx.moveTo(points[0].x, points[0].y)
      
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y)
      }
      ctx.stroke()

      // Draw trail points with fading opacity
      points.forEach((point, index) => {
        const opacity = (index / points.length) * 0.8
        const size = 3 + (index / points.length) * 3
        ctx.fillStyle = `rgba(0, 255, 255, ${opacity})`
        ctx.beginPath()
        ctx.arc(point.x, point.y, size, 0, Math.PI * 2)
        ctx.fill()
      })
    }

    // Draw current vehicle position
    if (currentPoint) {
      // Glow effect
      const glowGradient = ctx.createRadialGradient(
        currentPoint.x, currentPoint.y, 0,
        currentPoint.x, currentPoint.y, 20
      )
      glowGradient.addColorStop(0, 'rgba(0, 255, 255, 0.8)')
      glowGradient.addColorStop(1, 'rgba(0, 255, 255, 0)')
      ctx.fillStyle = glowGradient
      ctx.beginPath()
      ctx.arc(currentPoint.x, currentPoint.y, 20, 0, Math.PI * 2)
      ctx.fill()

      // Vehicle marker
      ctx.fillStyle = '#00ffff'
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(currentPoint.x, currentPoint.y, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      // Direction indicator
      ctx.strokeStyle = '#ffff00'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(currentPoint.x, currentPoint.y)
      ctx.lineTo(currentPoint.x + 15, currentPoint.y - 10)
      ctx.stroke()
    }

    // Draw start marker
    if (points.length > 0) {
      ctx.fillStyle = '#00ff00'
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.font = '16px monospace'
      ctx.beginPath()
      ctx.arc(points[0].x, points[0].y, 10, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#00ff00'
      ctx.fillText('START', points[0].x + 15, points[0].y + 5)
    }

    // Draw info
    ctx.fillStyle = '#00ffff'
    ctx.font = '14px monospace'
    ctx.fillText(`Points: ${points.length}`, 10, 20)
    if (currentPoint) {
      ctx.fillText(`X: ${currentPoint.x.toFixed(1)} Y: ${currentPoint.y.toFixed(1)}`, 10, 40)
    }

  }, [points, currentPoint])

  useEffect(() => {
    if (!isRunning) return

    let lastTime = Date.now()
    const animate = () => {
      const now = Date.now()
      const elapsed = now - lastTime

      if (elapsed > 50) { // Update every 50ms
        const canvas = canvasRef.current
        if (!canvas) return

        // Generate new point (replace with real data from socket)
        const newPoint: Point = {
          x: Math.min(canvas.width - 50, 50 + points.length * 5 + Math.random() * 20 - 10),
          y: canvas.height / 2 + Math.sin(points.length * 0.1) * 100 + Math.random() * 30 - 15,
          timestamp: now,
        }

        setPoints(prev => {
          const updated = [...prev, newPoint]
          // Keep last 200 points
          return updated.slice(-200)
        })
        setCurrentPoint(newPoint)

        // Notify parent
        onDataUpdate({
          points: points.length,
          current: newPoint,
        })

        lastTime = now
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isRunning, points, onDataUpdate])

  const handleReset = () => {
    setPoints([])
    setCurrentPoint(null)
  }

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: 'pixelated' }}
      />
      
      {/* Control overlay */}
      <div className="absolute top-4 right-4 flex gap-2">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleReset}
          className="px-4 py-2 bg-cyber-primary/20 hover:bg-cyber-primary/40 border border-cyber-primary rounded-lg text-sm font-mono transition-colors"
        >
          RESET
        </motion.button>
      </div>

      {/* Status indicator */}
      <div className="absolute bottom-4 left-4">
        <div className="flex items-center gap-2 px-3 py-2 bg-black/50 backdrop-blur-sm rounded-lg border border-cyber-primary/30">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-cyber-success animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-xs font-mono">
            {isRunning ? 'RECORDING' : 'STANDBY'}
          </span>
        </div>
      </div>
    </div>
  )
}
