'use client'

import { motion } from 'framer-motion'
import { PlayIcon, StopIcon, ArrowPathIcon } from '@heroicons/react/24/solid'

interface ControlPanelProps {
  modelSelected: boolean
  isRunning: boolean
  onStart: () => void
  onStop: () => void
}

export default function ControlPanel({ modelSelected, isRunning, onStart, onStop }: ControlPanelProps) {
  return (
    <div className="bg-cyber-card/50 backdrop-blur-xl rounded-2xl border border-cyber-secondary/30 p-6 shadow-cyber-lg">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
        <span className="text-cyber-secondary">⚡</span>
        Simulation Controls
      </h2>

      <div className="flex items-center gap-4">
        {/* Start Button */}
        <motion.button
          whileHover={{ scale: modelSelected && !isRunning ? 1.05 : 1 }}
          whileTap={{ scale: modelSelected && !isRunning ? 0.95 : 1 }}
          onClick={onStart}
          disabled={!modelSelected || isRunning}
          className={`flex-1 py-4 px-6 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all ${
            modelSelected && !isRunning
              ? 'bg-gradient-to-r from-cyber-success to-green-500 text-black shadow-cyber hover:shadow-cyber-lg'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          <PlayIcon className="w-6 h-6" />
          START SIMULATION
        </motion.button>

        {/* Stop Button */}
        <motion.button
          whileHover={{ scale: isRunning ? 1.05 : 1 }}
          whileTap={{ scale: isRunning ? 0.95 : 1 }}
          onClick={onStop}
          disabled={!isRunning}
          className={`flex-1 py-4 px-6 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all ${
            isRunning
              ? 'bg-gradient-to-r from-cyber-danger to-red-500 text-white shadow-cyber hover:shadow-cyber-lg'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          <StopIcon className="w-6 h-6" />
          STOP
        </motion.button>

        {/* Reset Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="py-4 px-6 rounded-xl font-bold text-lg bg-cyber-accent/20 border border-cyber-accent hover:bg-cyber-accent/30 transition-all"
        >
          <ArrowPathIcon className="w-6 h-6" />
        </motion.button>
      </div>

      {/* Status */}
      <div className="mt-6 p-4 bg-black/30 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Status:</span>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-cyber-success animate-pulse' : 'bg-gray-500'}`} />
            <span className={`font-mono font-bold ${isRunning ? 'text-cyber-success' : 'text-gray-500'}`}>
              {isRunning ? 'RUNNING' : modelSelected ? 'READY' : 'NO MODEL SELECTED'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
