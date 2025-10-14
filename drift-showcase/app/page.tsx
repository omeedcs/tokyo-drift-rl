'use client'

import { motion } from 'framer-motion'
import { Car, Brain, TrendingUp, Zap, Code, Award, ChevronDown } from 'lucide-react'
import ResultsChart from '@/components/ResultsChart'
import SimulationViewer from '@/components/SimulationViewer'
import NetworkArchitecture from '@/components/NetworkArchitecture'
import MathSection from '@/components/MathSection'
import MetricsGrid from '@/components/MetricsGrid'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 via-blue-900 to-purple-900 text-white">
      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20"></div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center z-10 px-4"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring" }}
            className="mb-8"
          >
            <Car className="w-24 h-24 mx-auto text-blue-400 animate-float" />
          </motion.div>
          
          <h1 className="text-7xl md:text-8xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
            Autonomous Drifting
          </h1>
          
          <p className="text-2xl md:text-3xl text-gray-300 mb-4">
            Deep Reinforcement Learning + Inverse Kinodynamics
          </p>
          
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12">
            Teaching vehicles to drift autonomously through the gate using SAC, IKD, and advanced control theory
          </p>
          
          <div className="flex gap-4 justify-center flex-wrap">
            <a href="#results" className="px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-all transform hover:scale-105">
              View Results
            </a>
            <a href="#simulation" className="px-8 py-4 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-all transform hover:scale-105">
              Watch Simulation
            </a>
          </div>
        </motion.div>
        
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="absolute bottom-8"
        >
          <ChevronDown className="w-8 h-8 text-gray-400" />
        </motion.div>
      </section>

      {/* Key Achievements */}
      <section className="py-20 px-4 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16"
          >
            Key Achievements
          </motion.h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { icon: Award, title: "IKD Success", value: "+2.3%", desc: "Improvement over baseline" },
              { icon: Zap, title: "Faster Execution", value: "51 steps", desc: "vs baseline 53 steps" },
              { icon: TrendingUp, title: "Success Rate", value: "100%", desc: "Consistent completion" },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="bg-gradient-to-br from-blue-900/50 to-purple-900/50 p-8 rounded-xl border border-blue-500/20 hover:border-blue-500/40 transition-all"
              >
                <item.icon className="w-12 h-12 text-blue-400 mb-4" />
                <h3 className="text-2xl font-bold mb-2">{item.title}</h3>
                <p className="text-4xl font-bold text-blue-400 mb-2">{item.value}</p>
                <p className="text-gray-400">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Results Comparison */}
      <section id="results" className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16"
          >
            Performance Results
          </motion.h2>
          
          <ResultsChart />
          
          <div className="mt-12">
            <MetricsGrid />
          </div>
        </div>
      </section>

      {/* Simulation Viewer */}
      <section id="simulation" className="py-20 px-4 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16"
          >
            Live Simulation
          </motion.h2>
          
          <SimulationViewer />
        </div>
      </section>

      {/* Neural Network Architecture */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16 flex items-center justify-center gap-4"
          >
            <Brain className="w-12 h-12 text-purple-400" />
            Neural Network Architecture
          </motion.h2>
          
          <NetworkArchitecture />
        </div>
      </section>

      {/* Mathematical Foundation */}
      <section className="py-20 px-4 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16"
          >
            Mathematical Foundation
          </motion.h2>
          
          <MathSection />
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold text-center mb-16 flex items-center justify-center gap-4"
          >
            <Code className="w-12 h-12 text-green-400" />
            Technology Stack
          </motion.h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { name: "PyTorch", desc: "Deep Learning", color: "from-orange-500 to-red-500" },
              { name: "SAC Algorithm", desc: "Soft Actor-Critic", color: "from-blue-500 to-cyan-500" },
              { name: "Gymnasium", desc: "RL Environment", color: "from-green-500 to-emerald-500" },
              { name: "NumPy/SciPy", desc: "Scientific Computing", color: "from-purple-500 to-pink-500" },
              { name: "Matplotlib", desc: "Visualization", color: "from-yellow-500 to-orange-500" },
              { name: "Jake's RL Algos", desc: "10+ Algorithms", color: "from-indigo-500 to-purple-500" },
              { name: "Custom IKD", desc: "Inverse Dynamics", color: "from-pink-500 to-rose-500" },
              { name: "Pygame", desc: "2D Simulation", color: "from-cyan-500 to-blue-500" },
            ].map((tech, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="p-6 rounded-xl bg-gray-800/50 border border-gray-700 hover:border-gray-500 transition-all"
              >
                <div className={`h-2 w-full rounded-full bg-gradient-to-r ${tech.color} mb-4`}></div>
                <h3 className="text-xl font-bold mb-2">{tech.name}</h3>
                <p className="text-gray-400">{tech.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Project Stats */}
      <section className="py-20 px-4 bg-gradient-to-b from-gray-900 to-black">
        <div className="max-w-7xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-5xl font-bold mb-16"
          >
            By the Numbers
          </motion.h2>
          
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { value: "15,000+", label: "Lines of Code" },
              { value: "20+", label: "Documentation Files" },
              { value: "15,900", label: "Training Samples" },
              { value: "100%", label: "Test Success Rate" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
              >
                <p className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
                  {stat.value}
                </p>
                <p className="text-xl text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 bg-black border-t border-gray-800">
        <div className="max-w-7xl mx-auto text-center text-gray-400">
          <p className="mb-4">Autonomous Vehicle Drifting Research Project</p>
          <p className="text-sm">Built with Next.js, React, and Framer Motion</p>
        </div>
      </footer>
    </main>
  )
}
