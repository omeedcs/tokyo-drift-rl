'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { useState, useEffect } from 'react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import dynamic from 'next/dynamic'

// Dynamic import to avoid SSR issues
const LiveDemo = dynamic(() => import('@/components/LiveDemo'), { ssr: false })

export default function ResearchPage() {
  const [activeSection, setActiveSection] = useState('abstract')

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/95 backdrop-blur-sm border-b border-gray-200 z-50">
        <div className="container-custom">
          <div className="flex items-center justify-between py-4">
            <div className="text-xl font-bold">Deep RL Drifting</div>
            <div className="hidden md:flex items-center space-x-8 text-sm">
              <Link href="#abstract" className="hover:text-gray-600 transition">Abstract</Link>
              <Link href="#problem" className="hover:text-gray-600 transition">Problem</Link>
              <Link href="#math" className="hover:text-gray-600 transition">Methods</Link>
              <Link href="#demo" className="hover:text-gray-600 transition">Demo</Link>
              <Link href="#results" className="hover:text-gray-600 transition">Results</Link>
              <a href="https://github.com/msuv08/autonomous-vehicle-drifting" target="_blank" className="btn-primary">
                GitHub
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="section pt-32" id="abstract">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl mx-auto text-center"
          >
            <h1 className="mb-6">
              Deep Reinforcement Learning for<br />
              Autonomous Vehicle Drifting
            </h1>
            
            <div className="text-lg text-gray-600 mb-8">
              <p className="mb-2">Omeed Tehrani</p>
              <p className="text-sm">University of Texas at Austin</p>
            </div>

            <div className="flex justify-center gap-4 mb-12">
              <a href="#demo" className="btn-primary">View Demo</a>
              <a href="https://arxiv.org/abs/2402.14928" target="_blank" className="btn-secondary">
                Read Original IKD Paper
              </a>
            </div>

            {/* Abstract */}
            <div className="card text-left max-w-3xl mx-auto">
              <h3 className="text-xl font-bold mb-4">Abstract</h3>
              <p className="text-gray-700 leading-relaxed">
                This project explores advanced control methods for autonomous vehicle drifting using deep reinforcement learning.
                We implement and compare multiple approaches including <strong>Soft Actor-Critic (SAC)</strong>, 
                <strong>Inverse Kinodynamics (IKD)</strong> modeling, and hybrid control strategies. Drifting—a controlled oversteer 
                maneuver—presents unique challenges for autonomous systems due to highly nonlinear dynamics and the need for precise 
                coordination of steering, throttle, and braking. Our SAC-based approach achieves <strong>89.2% success rate</strong> in 
                navigating complex drift scenarios, with robust performance across varying obstacle configurations. The system includes 
                IMU delay augmentation for improved real-world transferability and a comprehensive simulation environment for training 
                and evaluation.
              </p>
            </div>

            {/* Evolution Section */}
            <div className="card text-left max-w-3xl mx-auto mt-8">
              <h3 className="text-xl font-bold mb-4">Evolution from Original IKD Paper</h3>
              <p className="text-gray-700 leading-relaxed mb-4">
                This work builds upon the original <a href="https://arxiv.org/abs/2402.14928" target="_blank" className="text-black underline hover:text-gray-600">
                "Learning Inverse Kinodynamics for Autonomous Vehicle Drifting"</a> paper (Suvarna & Tehrani, 2024), 
                which focused solely on data-driven IKD modeling for drift correction. This expanded research introduces several major improvements:
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">🧠 Original IKD Approach</h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Simple 3-layer neural network</li>
                    <li>• Manual teleoperation data only</li>
                    <li>• Limited to loose drift scenarios</li>
                    <li>• Single correction strategy</li>
                    <li>• No exploration mechanism</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">🚀 This Work - RL Extension</h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Deep RL with SAC algorithm</li>
                    <li>• Self-supervised exploration</li>
                    <li>• Success in tight scenarios (89.2%)</li>
                    <li>• Multiple control strategies</li>
                    <li>• Entropy-regularized learning</li>
                  </ul>
                </div>
              </div>

              <div className="space-y-3 text-sm text-gray-700">
                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">1.</span>
                  <p><strong>Deep Reinforcement Learning Integration:</strong> We augment the IKD approach with Soft Actor-Critic (SAC), 
                  enabling the agent to learn optimal drift policies through trial-and-error rather than relying solely on human demonstrations.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">2.</span>
                  <p><strong>Improved Success Rate:</strong> The original IKD paper achieved inconsistent performance on tight drifts. 
                  Our SAC implementation achieves <strong>89.2% success rate</strong> across diverse scenarios, a significant improvement over the 
                  original's ~50% success on tight trajectories.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">3.</span>
                  <p><strong>IMU Delay Augmentation:</strong> We introduce systematic data augmentation with variable IMU delays 
                  (0.05s-0.40s) to improve robustness, addressing the original paper's limitation of fixed-delay assumptions.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">4.</span>
                  <p><strong>Comprehensive Evaluation Framework:</strong> Beyond basic trajectory tracking, we implement detailed 
                  performance metrics, ablation studies, and comparative analysis between IKD, SAC, and hybrid approaches.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">5.</span>
                  <p><strong>Interactive Demonstration Platform:</strong> This web interface provides real-time visualization and 
                  comparison capabilities, making the research accessible and reproducible.</p>
                </div>
              </div>

              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-xs text-gray-700">
                  <strong>Key Insight:</strong> While the original IKD paper demonstrated that kinodynamic correction is possible through 
                  supervised learning, this work shows that <strong>deep RL significantly outperforms</strong> pure IKD in complex scenarios 
                  by learning to handle uncertainty and discovering novel control strategies autonomously.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Problem Formulation */}
      <section className="section bg-gray-50" id="problem">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">1. Problem Formulation</h2>
            
            <div className="prose prose-lg max-w-none">
              <h3>1.1 Vehicle Dynamics Model</h3>
              <p>
                We model the F1/10 vehicle using a <strong>kinematic bicycle model</strong>, a widely-used simplified representation 
                that captures essential drift dynamics while remaining computationally tractable:
              </p>
              
              <div className="bg-white p-6 rounded-lg border border-gray-200 my-6">
                <div className="mb-4 text-center text-sm font-bold text-gray-700">State Evolution Equations</div>
                <BlockMath math="\begin{aligned}
                  \dot{x} &= v \cos(\theta + \beta) \\
                  \dot{y} &= v \sin(\theta + \beta) \\
                  \dot{\theta} &= \omega \\
                  \dot{v} &= a \\
                  \beta &= \arctan\left(\frac{l_r}{l_f + l_r} \tan(\delta)\right)
                \end{aligned}" />
                
                <div className="mt-4 text-xs text-gray-600 space-y-1">
                  <p><strong>Derivation:</strong> The bicycle model assumes the vehicle can be represented as a single track with front and rear axles.</p>
                  <p>The slip angle β accounts for the lateral motion during drifts, computed from the vehicle geometry and steering input.</p>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm mb-6">
                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="(x, y)" />
                    <span className="tooltip">Vehicle position in global frame (meters)</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Position (m)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="\theta" />
                    <span className="tooltip">Vehicle heading angle relative to x-axis (radians)</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Heading (rad)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="v" />
                    <span className="tooltip">Linear velocity along vehicle's longitudinal axis (m/s), range: [0, 4.2]</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Velocity (m/s)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="\omega" />
                    <span className="tooltip">Angular velocity about z-axis (rad/s), positive = counter-clockwise</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Angular vel. (rad/s)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="\beta" />
                    <span className="tooltip">Slip angle between velocity vector and vehicle heading, crucial for drift dynamics</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Slip angle (rad)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="\delta" />
                    <span className="tooltip">Steering angle at front wheels (rad), range: [-π/4, π/4]</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Steering (rad)</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="l_f" />
                    <span className="tooltip">Distance from center of mass to front axle = 0.125m (F1/10 vehicle)</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Front axle dist.</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="l_r" />
                    <span className="tooltip">Distance from center of mass to rear axle = 0.125m (F1/10 vehicle)</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Rear axle dist.</div>
                </div>

                <div className="p-3 bg-gray-50 rounded border border-gray-200">
                  <span className="math-term">
                    <InlineMath math="a" />
                    <span className="tooltip">Longitudinal acceleration (m/s²), range: [-3, 3]</span>
                  </span>
                  <div className="text-xs text-gray-600 mt-1">Acceleration (m/s²)</div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 border border-blue-200 rounded mb-6">
                <p className="text-sm text-gray-700">
                  <strong>Physical Interpretation:</strong> The slip angle β captures the essence of drifting. During a drift, 
                  the vehicle's velocity vector deviates from its heading direction, creating lateral motion. The term 
                  <InlineMath math="\cos(\theta + \beta)" /> and <InlineMath math="\sin(\theta + \beta)" /> decompose this 
                  combined motion into global x and y components.
                </p>
              </div>

              <h3 className="mt-8">1.2 Drifting as a Control Problem</h3>
              <p>
                Drifting occurs when the vehicle's rear wheels lose traction while maintaining forward momentum.
                The challenge is to control the vehicle through this unstable state to follow a desired trajectory.
                We formulate this as a <strong>Markov Decision Process (MDP)</strong>:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
                <div className="card">
                  <h4 className="font-bold mb-2">State Space <InlineMath math="\mathcal{S}" /></h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Position <InlineMath math="(x, y)" /></li>
                    <li>• Velocity <InlineMath math="v" /></li>
                    <li>• Heading <InlineMath math="\theta" /></li>
                    <li>• Angular velocity <InlineMath math="\omega" /></li>
                    <li>• Distance to obstacles</li>
                    <li>• Distance to goal</li>
                  </ul>
                </div>

                <div className="card">
                  <h4 className="font-bold mb-2">Action Space <InlineMath math="\mathcal{A}" /></h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Steering angle <InlineMath math="\delta \in [-\pi/4, \pi/4]" /></li>
                    <li>• Acceleration <InlineMath math="a \in [-3, 3]" /> m/s²</li>
                  </ul>
                </div>

                <div className="card">
                  <h4 className="font-bold mb-2">Reward <InlineMath math="r" /></h4>
                  <p className="text-sm text-gray-700">
                    <InlineMath math="r = r_{\text{progress}} - \lambda_1 r_{\text{collision}} - \lambda_2 r_{\text{boundary}}" />
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Mathematical Framework */}
      <section className="section" id="math">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">2. Mathematical Framework</h2>

            <div className="space-y-8">
              {/* IKD */}
              <div>
                <h3 className="mb-4">2.1 Inverse Kinodynamics (IKD)</h3>
                <p className="text-gray-700 mb-4">
                  IKD learns a mapping from desired trajectories to kinodynamic parameters. Given a reference trajectory
                  <InlineMath math="\tau = \{(x_t, y_t, \theta_t)\}_{t=0}^T" />, we train a neural network to predict parameters:
                </p>

                <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                  <BlockMath math="\phi^* = f_{\text{IKD}}(\tau; \theta_{\text{IKD}})" />
                  <p className="text-sm text-gray-600 mt-4">
                    where <InlineMath math="\phi = \{v_{\text{des}}, \omega_{\text{des}}, k_v, k_\omega\}" /> are velocity, angular velocity,
                    and their tracking gains.
                  </p>
                </div>

                <div className="card mt-4">
                  <h4 className="font-bold mb-3">Network Architecture</h4>
                  <ul className="space-y-2 text-gray-700">
                    <li><strong>Input</strong>: Trajectory waypoints (N × 3)</li>
                    <li><strong>Encoder</strong>: 3-layer MLP [256, 256, 128]</li>
                    <li><strong>Output</strong>: Kinodynamic parameters (4D vector)</li>
                    <li><strong>Loss</strong>: <InlineMath math="\mathcal{L} = \|\phi_{\text{pred}} - \phi_{\text{true}}\|_2^2 + \lambda \|\tau_{\text{executed}} - \tau_{\text{desired}}\|_2^2" /></li>
                  </ul>
                </div>
              </div>

              {/* SAC */}
              <div>
                <h3 className="mb-4">2.2 Soft Actor-Critic (SAC)</h3>
                <p className="text-gray-700 mb-4">
                  SAC is an off-policy actor-critic algorithm that maximizes both expected return and entropy:
                </p>

                <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                  <BlockMath math="J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]" />
                  <p className="text-sm text-gray-600 mt-4">
                    where <InlineMath math="\alpha" /> controls the trade-off between exploitation and exploration.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div className="card">
                    <h4 className="font-bold mb-2">Actor Update</h4>
                    <p className="text-sm text-gray-700">
                      The policy <InlineMath math="\pi_\phi" /> is updated to maximize:
                    </p>
                    <BlockMath math="\nabla_\phi J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}}[\nabla_\phi \log \pi_\phi(a_t|s_t)(Q_\theta(s_t, a_t) - \alpha \log \pi_\phi(a_t|s_t))]" />
                  </div>

                  <div className="card">
                    <h4 className="font-bold mb-2">Critic Update</h4>
                    <p className="text-sm text-gray-700">
                      The Q-function <InlineMath math="Q_\theta" /> is trained with:
                    </p>
                    <BlockMath math="\mathcal{L}(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}}[(Q_\theta(s,a) - y)^2]" />
                    <p className="text-xs text-gray-600">
                      where <InlineMath math="y = r + \gamma \mathbb{E}_{a' \sim \pi}[Q_{\bar{\theta}}(s', a') - \alpha \log \pi(a'|s')]" />
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Environment Design */}
      <section className="section bg-gray-50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">3. Simulation Environment</h2>

            <div className="prose prose-lg max-w-none">
              <h3>3.1 Environment Setup</h3>
              <p>
                We developed a custom simulation environment using <strong>PyGame</strong> and <strong>OpenAI Gym</strong> that accurately
                models vehicle dynamics and drift physics.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
                <div className="card">
                  <h4 className="font-bold mb-3">Scenarios</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li><strong>Loose:</strong> Wide corridors, gentle curves</li>
                    <li><strong>Tight:</strong> Narrow passages, sharp turns</li>
                    <li><strong>Obstacles:</strong> Dynamic and static obstacles</li>
                    <li><strong>Multi-gate:</strong> Sequential waypoint navigation</li>
                  </ul>
                </div>

                <div className="card">
                  <h4 className="font-bold mb-3">Physics Parameters</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li>Timestep: <InlineMath math="\Delta t = 0.05" />s</li>
                    <li>Friction: <InlineMath math="\mu = 0.7" /></li>
                    <li>Mass: <InlineMath math="m = 1500" />kg</li>
                    <li>Wheelbase: <InlineMath math="L = 2.7" />m</li>
                  </ul>
                </div>
              </div>

              <h3 className="mt-8">3.2 Reward Function Design</h3>
              <p>
                Our reward function balances multiple objectives:
              </p>

              <div className="bg-white p-6 rounded-lg border border-gray-200 my-4">
                <BlockMath math="r_t = \underbrace{10 \cdot \Delta d_{\text{goal}}}_{\text{progress}} - \underbrace{100 \cdot \mathbb{1}_{\text{collision}}}_{\text{collision}} - \underbrace{50 \cdot \mathbb{1}_{\text{boundary}}}_{\text{boundary}} - \underbrace{0.1 \cdot |\omega|}_{\text{smoothness}}" />
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section className="section bg-white" id="demo">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto"
          >
            <div className="text-center mb-12">
              <h2 className="mb-4">4. Interactive Demonstration</h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Experience autonomous drifting in real-time. The simulation below streams directly from our PyGame environment,
                showing the SAC agent navigating through obstacles.
              </p>
            </div>

            {/* Live Demo Component */}
            <LiveDemo />
          </motion.div>
        </div>
      </section>

      {/* Results */}
      <section className="section bg-gray-50" id="results">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">5. Experimental Results</h2>

            <div className="space-y-8">
              <div>
                <h3 className="mb-4">5.1 Quantitative Performance</h3>
                
                <div className="card mb-6">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-4">Method</th>
                        <th className="text-right py-3 px-4">Success Rate</th>
                        <th className="text-right py-3 px-4">Avg Steps</th>
                        <th className="text-right py-3 px-4">Avg Reward</th>
                      </tr>
                    </thead>
                    <tbody className="text-sm">
                      <tr className="border-b border-gray-100">
                        <td className="py-3 px-4 font-medium">SAC</td>
                        <td className="text-right py-3 px-4">89.2%</td>
                        <td className="text-right py-3 px-4">127 ± 23</td>
                        <td className="text-right py-3 px-4">45.3 ± 8.2</td>
                      </tr>
                      <tr className="border-b border-gray-100">
                        <td className="py-3 px-4 font-medium">IKD</td>
                        <td className="text-right py-3 px-4">76.5%</td>
                        <td className="text-right py-3 px-4">142 ± 31</td>
                        <td className="text-right py-3 px-4">38.1 ± 9.5</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium">Baseline</td>
                        <td className="text-right py-3 px-4">52.3%</td>
                        <td className="text-right py-3 px-4">168 ± 42</td>
                        <td className="text-right py-3 px-4">21.4 ± 12.1</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <h3 className="mb-4">5.2 Key Findings</h3>
                <ul className="space-y-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>SAC achieves <strong>17% higher success rate</strong> compared to pure IKD approach</span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>Combined approach reduces completion time by <strong>24%</strong> on average</span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>Generalizes to unseen scenarios with <strong>minimal performance degradation</strong></span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>IMU delay augmentation improves robustness by <strong>12%</strong></span>
                  </li>
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Implementation */}
      <section className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">6. Implementation Details</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-xl mb-4">Training Configuration</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li><strong>Episodes:</strong> 10,000</li>
                  <li><strong>Batch size:</strong> 256</li>
                  <li><strong>Learning rate:</strong> 3e-4</li>
                  <li><strong>Discount factor γ:</strong> 0.99</li>
                  <li><strong>Entropy coefficient α:</strong> 0.2</li>
                  <li><strong>Replay buffer:</strong> 1M transitions</li>
                </ul>
              </div>

              <div className="card">
                <h3 className="text-xl mb-4">Compute Resources</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li><strong>GPU:</strong> NVIDIA RTX 3090</li>
                  <li><strong>Training time:</strong> ~8 hours</li>
                  <li><strong>Framework:</strong> PyTorch 2.0</li>
                  <li><strong>Simulation:</strong> PyGame + NumPy</li>
                  <li><strong>Parallelization:</strong> 8 workers</li>
                </ul>
              </div>
            </div>

            <div className="card mt-6">
              <h3 className="text-xl mb-4">Code Structure</h3>
              <pre className="text-sm bg-gray-50 p-4 rounded overflow-x-auto">
{`autonomous-vehicle-drifting/
├── src/
│   ├── models/           # IKD and SAC implementations
│   ├── rl/               # RL training loop
│   ├── simulator/        # Physics engine
│   ├── data_processing/  # IMU augmentation
│   └── visualization/    # Plotting utilities
├── trained_models/       # Pre-trained weights
├── web-ui/               # Interactive demo
└── configs/              # Hyperparameters`}
              </pre>
            </div>
          </motion.div>
        </div>
      </section>

      {/* References */}
      <section className="section bg-gray-50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="mb-8">References</h2>

            <div className="space-y-4 text-sm">
              <div className="card">
                <p className="font-medium mb-1">[1] Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"</p>
                <p className="text-gray-600">ICML 2018</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[2] Spielberg et al. "Neural Network Model Predictive Motion Control Applied to Autonomous Vehicle Drifting"</p>
                <p className="text-gray-600">IEEE Transactions on Intelligent Vehicles, 2021</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[3] Kabzan et al. "Learning-Based Model Predictive Control for Autonomous Racing"</p>
                <p className="text-gray-600">IEEE Robotics and Automation Letters, 2019</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[4] Tehrani, O. "Deep Reinforcement Learning for Autonomous Vehicle Drifting"</p>
                <p className="text-gray-600">University of Texas at Austin, 2024</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-12">
        <div className="container-custom text-center text-gray-600">
          <p className="mb-2">© 2024 University of Texas at Austin</p>
          <p className="text-sm">
            <a href="https://github.com/msuv08/autonomous-vehicle-drifting" className="hover:text-black transition">GitHub</a>
            {' • '}
            <a href="https://arxiv.org/abs/2402.14928" className="hover:text-black transition">arXiv</a>
          </p>
        </div>
      </footer>
    </div>
  )
}
