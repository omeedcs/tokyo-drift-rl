'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

interface MetricsPanelProps {
  isRunning: boolean
  data: any
}

export default function MetricsPanel({ isRunning, data }: MetricsPanelProps) {
  const [metrics, setMetrics] = useState({
    step: 0,
    reward: 0,
    totalReward: 0,
    velocity: 0,
    angularVelocity: 0,
    position: { x: 0, y: 0 },
  })

  useEffect(() => {
    if (data) {
      setMetrics(prev => ({
        ...prev,
        ...data,
        totalReward: (prev.totalReward || 0) + (data.reward || 0),
      }))
    }
  }, [data])

  const MetricCard = ({ label, value, unit, color }: any) => (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className="bg-black/30 rounded-lg p-4 border border-gray-700 hover:border-cyber-primary/50 transition-all"
    >
      <div className="text-gray-400 text-sm mb-1">{label}</div>
      <div className={`text-2xl font-bold font-mono ${color}`}>
        {typeof value === 'number' ? value.toFixed(2) : value}
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
      </div>
    </motion.div>
  )

  return (
    <div className="h-full space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="Step"
          value={metrics.step}
          color="text-cyber-primary"
        />
        <MetricCard
          label="Total Reward"
          value={metrics.totalReward}
          color="text-cyber-secondary"
        />
        <MetricCard
          label="Velocity"
          value={metrics.velocity}
          unit="m/s"
          color="text-cyber-accent"
        />
        <MetricCard
          label="Angular Vel"
          value={metrics.angularVelocity}
          unit="rad/s"
          color="text-cyber-success"
        />
      </div>

      <div className="bg-black/30 rounded-lg p-4 border border-gray-700">
        <div className="text-gray-400 text-sm mb-3">Position</div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-gray-500 text-xs">X</div>
            <div className="text-xl font-mono font-bold text-cyber-primary">
              {metrics.position.x.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Y</div>
            <div className="text-xl font-mono font-bold text-cyber-secondary">
              {metrics.position.y.toFixed(2)}
            </div>
          </div>
        </div>
      </div>

      {!isRunning && (
        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">📊</div>
            <p>Start simulation to view metrics</p>
          </div>
        </div>
      )}
    </div>
  )
}
