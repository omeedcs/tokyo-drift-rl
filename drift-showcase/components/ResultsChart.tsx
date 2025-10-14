'use client'

import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const data = [
  {
    method: 'Baseline',
    reward: -76.88,
    steps: 53,
    successRate: 100,
  },
  {
    method: 'IKD',
    reward: -75.12,
    steps: 51,
    successRate: 100,
  },
  {
    method: 'SAC',
    reward: -70.5,
    steps: 48,
    successRate: 95,
  },
]

export default function ResultsChart() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="bg-gray-800/50 p-8 rounded-xl border border-gray-700"
    >
      <h3 className="text-3xl font-bold mb-8 text-center">Method Comparison</h3>
      
      <div className="grid md:grid-cols-3 gap-8">
        {/* Average Reward */}
        <div>
          <h4 className="text-xl font-semibold mb-4 text-blue-400">Average Reward</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="method" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="reward" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Steps to Completion */}
        <div>
          <h4 className="text-xl font-semibold mb-4 text-purple-400">Steps to Complete</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="method" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="steps" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Success Rate */}
        <div>
          <h4 className="text-xl font-semibold mb-4 text-green-400">Success Rate (%)</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="method" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="successRate" fill="#10B981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Comparison Table */}
      <div className="mt-12 overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="pb-4 text-xl">Method</th>
              <th className="pb-4 text-xl">Avg Reward</th>
              <th className="pb-4 text-xl">Success Rate</th>
              <th className="pb-4 text-xl">Avg Steps</th>
              <th className="pb-4 text-xl">Improvement</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-800">
              <td className="py-4 font-semibold">Baseline</td>
              <td className="py-4">-76.88</td>
              <td className="py-4">100.0%</td>
              <td className="py-4">53.0</td>
              <td className="py-4 text-gray-400">+0.0%</td>
            </tr>
            <tr className="border-b border-gray-800">
              <td className="py-4 font-semibold">IKD</td>
              <td className="py-4">-75.12</td>
              <td className="py-4">100.0%</td>
              <td className="py-4">51.0</td>
              <td className="py-4 text-green-400">+2.3%</td>
            </tr>
            <tr>
              <td className="py-4 font-semibold">SAC (Expected)</td>
              <td className="py-4">-70.5</td>
              <td className="py-4">95.0%</td>
              <td className="py-4">48.0</td>
              <td className="py-4 text-green-400">+8.3%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </motion.div>
  )
}
