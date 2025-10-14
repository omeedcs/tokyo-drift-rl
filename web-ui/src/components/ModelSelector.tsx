'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface ModelSelectorProps {
  onModelSelect: (model: string) => void
  selectedModel: string | null
}

export default function ModelSelector({ onModelSelect, selectedModel }: ModelSelectorProps) {
  const [ikdModels, setIkdModels] = useState<string[]>([])
  const [sacModels, setSacModels] = useState<string[]>([])
  const [modelType, setModelType] = useState<'ikd' | 'sac'>('sac')

  useEffect(() => {
    // Fetch available models from server
    fetch('http://localhost:5001/api/models')
      .then(res => res.json())
      .then(data => {
        setIkdModels(data.ikd || [])
        setSacModels(data.sac || [])
      })
      .catch(err => {
        console.error('Failed to fetch models:', err)
        // Fallback data
        setIkdModels(['ikd_final', 'ikd_corrected', 'ikd_tracking'])
        setSacModels(['sac_loose_0', 'sac_loose_1', 'sac_loose_2'])
      })
  }, [])

  const models = modelType === 'ikd' ? ikdModels : sacModels

  return (
    <div className="bg-cyber-card/50 backdrop-blur-xl rounded-2xl border border-cyber-primary/30 p-6 shadow-cyber-lg">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
        <span className="text-cyber-primary">🤖</span>
        Model Selection
      </h2>

      {/* Model Type Selector */}
      <div className="flex gap-4 mb-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setModelType('ikd')}
          className={`flex-1 py-3 rounded-lg font-mono font-bold transition-all ${
            modelType === 'ikd'
              ? 'bg-gradient-to-r from-cyber-primary to-cyan-500 text-black shadow-cyber'
              : 'bg-cyber-card border border-cyber-primary/30 text-gray-400 hover:text-white'
          }`}
        >
          IKD Models
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setModelType('sac')}
          className={`flex-1 py-3 rounded-lg font-mono font-bold transition-all ${
            modelType === 'sac'
              ? 'bg-gradient-to-r from-cyber-secondary to-purple-500 text-black shadow-cyber'
              : 'bg-cyber-card border border-cyber-secondary/30 text-gray-400 hover:text-white'
          }`}
        >
          SAC Models
        </motion.button>
      </div>

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((model) => (
          <motion.button
            key={model}
            whileHover={{ scale: 1.05, y: -5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onModelSelect(model)}
            className={`p-4 rounded-xl border-2 transition-all ${
              selectedModel === model
                ? 'bg-gradient-to-br from-cyber-primary/20 to-cyber-secondary/20 border-cyber-primary shadow-cyber-lg'
                : 'bg-cyber-card/30 border-gray-700 hover:border-cyber-primary/50'
            }`}
          >
            <div className="text-sm font-mono text-gray-400 mb-2">
              {modelType.toUpperCase()}
            </div>
            <div className="font-bold text-lg">{model}</div>
            {selectedModel === model && (
              <div className="mt-2 text-cyber-success text-sm">✓ Selected</div>
            )}
          </motion.button>
        ))}
      </div>

      {/* Selected Model Info */}
      {selectedModel && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-cyber-primary/10 rounded-lg border border-cyber-primary/30"
        >
          <div className="text-sm text-gray-400 mb-1">Active Model:</div>
          <div className="text-xl font-bold text-cyber-primary font-mono">
            {selectedModel}
          </div>
        </motion.div>
      )}
    </div>
  )
}
