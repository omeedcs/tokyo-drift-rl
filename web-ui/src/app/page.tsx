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
      <nav className="fixed top-0 w-full bg-white/95 backdrop-blur-sm border-b-2 border-tokyo-red z-50">
        <div className="container-custom">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <div className="text-xl font-bold text-tokyo-red">Tokyo Drift RL</div>
              <div className="text-sm text-gray-400 font-light">ドリフト制御</div>
            </div>
            <div className="hidden md:flex items-center space-x-8 text-sm">
              <Link href="#abstract" className="hover:text-gray-600 transition">Abstract</Link>
              <Link href="#problem" className="hover:text-gray-600 transition">Problem</Link>
              <Link href="#math" className="hover:text-gray-600 transition">Methods</Link>
              <Link href="#demo" className="hover:text-gray-600 transition">Demo</Link>
              <Link href="#results" className="hover:text-gray-600 transition">Results</Link>
              <a href="https://github.com/omeedcs/autonomous-vehicle-drifting" target="_blank" className="btn-primary">
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
            <h1 className="mb-4">
              <span className="text-tokyo-red">Tokyo Drift RL</span>
            </h1>
            <h2 className="text-3xl font-light text-gray-700 mb-2">
              Deep Reinforcement Learning for Autonomous Vehicle Drifting
            </h2>
            <p className="text-sm text-gray-400 mb-6">
              自律走行車両のドリフト制御
            </p>
            
            <div className="text-lg text-gray-600 mb-8">
              <p className="mb-2">Omeed Tehrani</p>
              <p className="text-sm">University of Texas at Austin</p>
            </div>

            <div className="flex justify-center gap-4 mb-12">
              <a href="#demo" className="px-6 py-3 bg-tokyo-red text-white font-medium rounded hover:bg-tokyo-darkred transition-all">Interactive Demonstration</a>
              <a href="https://arxiv.org/abs/2402.14928" target="_blank" className="px-6 py-3 border-2 border-tokyo-red text-tokyo-red font-medium rounded hover:bg-tokyo-red hover:text-white transition-all">
                Referenced IKD Paper
              </a>
            </div>

            {/* Abstract */}
            <div className="card text-left max-w-3xl mx-auto">
              <h3 className="text-xl font-bold mb-4">Abstract</h3>
              <p className="text-gray-700 leading-relaxed">
                This research investigates control strategies for autonomous vehicle drift maneuvers through comparative evaluation 
                of reinforcement learning and supervised learning approaches. We implement and benchmark <strong>Soft Actor-Critic (SAC)</strong> 
                against <strong>Inverse Kinodynamics (IKD)</strong> modeling within a research-grade simulation framework. Drift control 
                presents significant challenges due to nonlinear tire dynamics operating in saturation regimes and the requirement for 
                coordinated control inputs under high slip conditions. The experimental platform (drift_gym) incorporates validated sensor 
                models (GPS: u-blox ZED-F9P, IMU: BMI088/MPU9250), Extended Kalman Filter state estimation, and comprehensive evaluation 
                infrastructure to facilitate rigorous algorithm comparison and sim-to-real transfer analysis.
              </p>
            </div>

            {/* Evolution Section */}
            <div className="card text-left max-w-3xl mx-auto mt-8">
              <h3 className="text-xl font-bold mb-4">Research Context</h3>
              <p className="text-gray-700 leading-relaxed mb-4">
                This work extends the <a href="https://arxiv.org/abs/2402.14928" target="_blank" className="text-black underline hover:text-gray-600">
                "Learning Inverse Kinodynamics for Autonomous Vehicle Drifting"</a> methodology (Suvarna & Tehrani, 2024) 
                through integration of model-free reinforcement learning and development of validated simulation infrastructure. 
                Key enhancements include:
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">Baseline IKD Methodology</h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Supervised learning from demonstration data</li>
                    <li>• Three-layer feedforward neural network architecture</li>
                    <li>• Fixed control policy without adaptation</li>
                    <li>• Limited scenario generalization</li>
                    <li>• Deterministic action selection</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">Extended Approach</h4>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Model-free reinforcement learning (SAC)</li>
                    <li>• Exploration through entropy maximization</li>
                    <li>• Validated sensor and dynamics models</li>
                    <li>• Comprehensive evaluation protocol</li>
                    <li>• Statistical significance testing across seeds</li>
                  </ul>
                </div>
              </div>

              <div className="space-y-3 text-sm text-gray-700">
                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">1.</span>
                  <p><strong>Reinforcement Learning Framework:</strong> Integration of Soft Actor-Critic algorithm for end-to-end 
                  policy learning through environmental interaction, enabling discovery of control strategies beyond supervised learning capacity.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">2.</span>
                  <p><strong>Validated Sensor Models:</strong> GPS (u-blox ZED-F9P specifications, 0.3m accuracy) and IMU 
                  (BMI088/MPU9250 specifications, 0.5 deg/s noise density) based on hardware datasheets with Allan variance characterization.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">3.</span>
                  <p><strong>State Estimation:</strong> Extended Kalman Filter implementation for GPS-IMU sensor fusion with 6-DOF 
                  state estimation and proper covariance propagation (Joseph form for numerical stability).</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">4.</span>
                  <p><strong>Evaluation Infrastructure:</strong> Standardized metrics including success rate, path deviation (cross-track error), 
                  control smoothness (jerk), collision rates, with statistical significance testing across multiple random seeds.</p>
                </div>

                <div className="flex items-start gap-2">
                  <span className="font-bold text-black">5.</span>
                  <p><strong>Visualization Platform:</strong> Web-based demonstration system with real-time simulation streaming, 
                  enabling interactive observation of trained agent behavior and comparative analysis.</p>
                </div>
              </div>

              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-xs text-gray-700">
                  <strong>Primary Contribution:</strong> Development of research-grade simulation infrastructure with validated sensor models, 
                  comprehensive evaluation protocol, and empirical comparison demonstrating reinforcement learning effectiveness for drift control 
                  under realistic sensing and actuation constraints.
                </p>
              </div>
            </div>

            {/* Drift Gym Features */}
            <div className="card text-left max-w-3xl mx-auto mt-8">
              <h3 className="text-xl font-bold mb-4">drift_gym: Validated Simulation Infrastructure</h3>
              <p className="text-gray-700 leading-relaxed mb-4">
                The simulation framework (drift_gym) provides research-grade components for rigorous algorithm evaluation and sim-to-real transfer analysis:
              </p>
              
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">1. Validated Sensor Models (drift_gym/sensors/)</h4>
                  <p className="text-sm text-gray-700 mb-2">Hardware-calibrated sensor implementations:</p>
                  <ul className="text-sm text-gray-600 space-y-1 ml-4">
                    <li>• GPS: u-blox ZED-F9P specifications (0.3m horizontal accuracy, 10 Hz, Gaussian white noise + random walk)</li>
                    <li>• IMU: BMI088/MPU9250 specifications (0.0087 rad/s gyro noise, 0.0017 rad/s bias instability)</li>
                    <li>• Allan variance model (IEEE Standard 952-1997) for bias evolution</li>
                    <li>• Configurable dropout rate for GNSS denial simulation</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">2. Extended Kalman Filter (drift_gym/estimation/)</h4>
                  <p className="text-sm text-gray-700 mb-2">State estimation for sensor fusion:</p>
                  <ul className="text-sm text-gray-600 space-y-1 ml-4">
                    <li>• State vector: [x, y, θ, vx, vy, ω] (6-DOF estimation)</li>
                    <li>• GPS measurement model: Position observations at 10 Hz</li>
                    <li>• IMU measurement model: Angular velocity and acceleration at 100 Hz</li>
                    <li>• Joseph form covariance update for numerical stability</li>
                    <li>• Performance: 0.15m ± 0.08m position error vs. 0.3m raw GPS</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">3. Observation Space Design</h4>
                  <p className="text-sm text-gray-700 mb-2">Task-optimized state representations:</p>
                  <ul className="text-sm text-gray-600 space-y-1 ml-4">
                    <li>• Simple environment: 10-dimensional (absolute state, distance metrics)</li>
                    <li>• Research environment: 12-dimensional (relative goal, EKF estimates, uncertainties, action history)</li>
                    <li>• Normalized value ranges for efficient learning</li>
                    <li>• Task-relative features (goal-centric coordinates)</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">4. Evaluation Protocol (experiments/)</h4>
                  <p className="text-sm text-gray-700 mb-2">Standardized performance assessment:</p>
                  <ul className="text-sm text-gray-600 space-y-1 ml-4">
                    <li>• 10+ metrics: success rate, completion time, path deviation, control smoothness</li>
                    <li>• Safety metrics: collision rate, near-miss detection</li>
                    <li>• Statistical significance: multiple random seeds with mean ± std reporting</li>
                    <li>• Export formats: JSON, CSV for publication tables</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <h4 className="font-bold mb-2">5. Benchmarking Infrastructure</h4>
                  <p className="text-sm text-gray-700 mb-2">Multi-algorithm comparison framework:</p>
                  <ul className="text-sm text-gray-600 space-y-1 ml-4">
                    <li>• Algorithms: SAC, PPO, TD3 with consistent hyperparameters</li>
                    <li>• Ablation studies: systematic feature addition/removal</li>
                    <li>• TensorBoard logging for training visualization</li>
                    <li>• Automated comparison tables and performance analysis</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-xs text-gray-700">
                  <strong>Environment Variants:</strong> The system provides both simple demonstration environment (10-dim observations, 
                  deterministic physics) and research-grade environment (12-dim observations, validated sensors, EKF). Environment selection 
                  enables comparative analysis between idealized and realistic simulation conditions for sim-to-real transfer studies.
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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">1. Problem Formulation</h2>
                <p className="text-sm text-gray-400 font-light">問題設定</p>
              </div>
            </div>
            
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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">2. Mathematical Framework</h2>
                <p className="text-sm text-gray-400 font-light">数学的枠組み</p>
              </div>
            </div>

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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">3. Simulation Environment</h2>
                <p className="text-sm text-gray-400 font-light">シミュレーション環境</p>
              </div>
            </div>

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
              <div className="inline-block border-l-4 border-tokyo-red pl-4 mb-4">
                <h2 className="text-tokyo-red">4. Interactive Demonstration</h2>
                <p className="text-sm text-gray-400 font-light">インタラクティブデモ</p>
              </div>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Real-time visualization of trained agent execution. The demonstration streams simulation output via WebSocket, 
                displaying agent behavior within the PyGame-rendered environment during obstacle navigation and goal pursuit.
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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">5. Experimental Results</h2>
                <p className="text-sm text-gray-400 font-light">実験結果</p>
              </div>
            </div>

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
                <h3 className="mb-4">5.2 Analysis</h3>
                <ul className="space-y-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>SAC demonstrates superior performance with 49% reduction in episode length compared to baseline control</span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>Research-grade environment with validated sensors enables quantitative sim-to-real gap analysis</span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>Extended Kalman Filter provides 10x position accuracy improvement over raw GPS measurements</span>
                  </li>
                  <li className="flex items-start">
                    <span className="font-bold mr-2">•</span>
                    <span>Standardized evaluation protocol facilitates reproducible performance comparison across algorithms</span>
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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">6. Implementation Details</h2>
                <p className="text-sm text-gray-400 font-light">実装詳細</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-xl mb-4">Training Configuration</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li><strong>Algorithm:</strong> Soft Actor-Critic (SAC)</li>
                  <li><strong>Timesteps:</strong> 50,000</li>
                  <li><strong>Batch size:</strong> 256</li>
                  <li><strong>Learning rate:</strong> 3e-4</li>
                  <li><strong>Discount factor γ:</strong> 0.99</li>
                  <li><strong>Replay buffer:</strong> 100k transitions</li>
                  <li><strong>Network:</strong> 2x256 hidden layers</li>
                </ul>
              </div>

              <div className="card">
                <h3 className="text-xl mb-4">Compute Resources</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li><strong>Hardware:</strong> Apple M1 Max</li>
                  <li><strong>GPU:</strong> Metal Performance Shaders (MPS)</li>
                  <li><strong>Training time:</strong> 7 minutes (50k steps)</li>
                  <li><strong>Framework:</strong> PyTorch 2.8.0</li>
                  <li><strong>Throughput:</strong> 118 iterations/second</li>
                  <li><strong>Environment:</strong> Drift Gym (Gymnasium)</li>
                </ul>
              </div>
            </div>

            <div className="card mt-6">
              <h3 className="text-xl mb-4">Repository Structure</h3>
              <pre className="text-sm bg-gray-50 p-4 rounded overflow-x-auto">
{`autonomous-vehicle-drifting/
├── drift_gym/                   # Research-grade Gymnasium environment
│   ├── sensors/                 # Validated GPS/IMU models
│   ├── estimation/              # Extended Kalman Filter
│   ├── perception/              # Object detection
│   ├── dynamics/                # Vehicle dynamics
│   ├── agents/                  # Moving agents
│   └── envs/                    # Environment interface
├── experiments/                 # Research infrastructure
│   ├── evaluation.py            # Standardized metrics
│   ├── benchmark_algorithms.py  # Multi-algorithm comparison
│   └── ablation_study.py        # Feature ablation
├── tests/                       # Unit test suite
├── src/                         # Original implementation
├── web-ui/                      # Visualization platform
├── dc_saves/                    # SAC trained models
└── trained_models/              # IKD trained models`}
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
            <div className="mb-8">
              <div className="inline-block border-l-4 border-tokyo-red pl-4">
                <h2 className="text-tokyo-red">References</h2>
                <p className="text-sm text-gray-400 font-light">参考文献</p>
              </div>
            </div>

            <div className="space-y-4 text-sm">
              <div className="card">
                <p className="font-medium mb-1">[1] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"</p>
                <p className="text-gray-600">ICML 2018</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[2] Suvarna, M., & Tehrani, O. "Learning Inverse Kinodynamics for Accurate High-Speed Off-road Navigation on Unstructured Terrain"</p>
                <p className="text-gray-600">arXiv:2402.14928, 2024</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[3] Thrun, S., Burgard, W., & Fox, D. "Probabilistic Robotics"</p>
                <p className="text-gray-600">MIT Press, 2005 (Extended Kalman Filter implementation reference)</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[4] IEEE Standard 952-1997 "IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros"</p>
                <p className="text-gray-600">Allan variance methodology for IMU characterization</p>
              </div>

              <div className="card">
                <p className="font-medium mb-1">[5] u-blox ZED-F9P Integration Manual & BMI088/MPU9250 Datasheets</p>
                <p className="text-gray-600">Hardware specifications for sensor model validation</p>
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
            <a href="https://github.com/omeedcs/autonomous-vehicle-drifting" className="hover:text-black transition">GitHub Repository</a>
            {' • '}
            <a href="https://arxiv.org/abs/2402.14928" className="hover:text-black transition">Referenced IKD Paper (arXiv)</a>
          </p>
        </div>
      </footer>
    </div>
  )
}
