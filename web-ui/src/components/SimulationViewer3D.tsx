'use client'

import { useRef, useEffect, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, PerspectiveCamera, Trail, Float } from '@react-three/drei'
import * as THREE from 'three'

interface Vehicle3DProps {
  position: [number, number, number]
  rotation: number
  isActive: boolean
}

function Vehicle3D({ position, rotation, isActive }: Vehicle3DProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (meshRef.current && isActive) {
      // Pulse effect when active
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1
      meshRef.current.scale.set(scale, scale, scale)
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <Trail
        width={3}
        length={20}
        color={new THREE.Color(0x00ffff)}
        attenuation={(t) => t * t}
      >
        <group position={position} rotation={[0, rotation, 0]}>
          {/* Vehicle Body */}
          <mesh ref={meshRef} castShadow>
            <boxGeometry args={[0.6, 0.3, 1.2]} />
            <meshStandardMaterial
              color={isActive ? "#00ffff" : "#ffffff"}
              emissive={isActive ? "#00ffff" : "#000000"}
              emissiveIntensity={isActive ? 0.5 : 0}
              metalness={0.8}
              roughness={0.2}
            />
          </mesh>
          
          {/* Front indicator */}
          <mesh position={[0, 0.2, 0.7]} castShadow>
            <coneGeometry args={[0.2, 0.4, 3]} />
            <meshStandardMaterial
              color="#ffff00"
              emissive="#ffff00"
              emissiveIntensity={1}
            />
          </mesh>
          
          {/* Wheels */}
          {[
            [-0.3, -0.15, 0.4],
            [0.3, -0.15, 0.4],
            [-0.3, -0.15, -0.4],
            [0.3, -0.15, -0.4],
          ].map((pos, i) => (
            <mesh key={i} position={pos as [number, number, number]} castShadow>
              <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
              <meshStandardMaterial color="#333333" />
            </mesh>
          ))}
        </group>
      </Trail>
    </Float>
  )
}

interface ObstacleProps {
  position: [number, number, number]
  radius: number
}

function Obstacle({ position, radius }: ObstacleProps) {
  return (
    <mesh position={position} castShadow>
      <cylinderGeometry args={[radius, radius, 1, 32]} />
      <meshStandardMaterial
        color="#ff0000"
        emissive="#ff0000"
        emissiveIntensity={0.3}
        transparent
        opacity={0.6}
      />
    </mesh>
  )
}

interface GoalGateProps {
  position: [number, number, number]
  width: number
}

function GoalGate({ position, width }: GoalGateProps) {
  return (
    <group position={position}>
      {/* Left post */}
      <mesh position={[0, 1, -width/2]} castShadow>
        <cylinderGeometry args={[0.1, 0.1, 2, 16]} />
        <meshStandardMaterial
          color="#00ff00"
          emissive="#00ff00"
          emissiveIntensity={0.8}
        />
      </mesh>
      
      {/* Right post */}
      <mesh position={[0, 1, width/2]} castShadow>
        <cylinderGeometry args={[0.1, 0.1, 2, 16]} />
        <meshStandardMaterial
          color="#00ff00"
          emissive="#00ff00"
          emissiveIntensity={0.8}
        />
      </mesh>
      
      {/* Top bar */}
      <mesh position={[0, 2, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, width, 16]} />
        <meshStandardMaterial
          color="#00ff00"
          emissive="#00ff00"
          emissiveIntensity={0.5}
        />
      </mesh>
    </group>
  )
}

function TrajectoryPath({ points }: { points: [number, number, number][] }) {
  if (points.length < 2) return null

  const curve = new THREE.CatmullRomCurve3(
    points.map(p => new THREE.Vector3(...p))
  )
  const tubeGeometry = new THREE.TubeGeometry(curve, points.length * 2, 0.05, 8, false)

  return (
    <mesh geometry={tubeGeometry}>
      <meshBasicMaterial
        color="#00ffff"
        transparent
        opacity={0.6}
      />
    </mesh>
  )
}

interface SimulationViewer3DProps {
  isRunning: boolean
  trajectoryData: any
}

export default function SimulationViewer3D({ isRunning, trajectoryData }: SimulationViewer3DProps) {
  const [vehiclePos, setVehiclePos] = useState<[number, number, number]>([0, 0, 0])
  const [vehicleRot, setVehicleRot] = useState(0)
  const [trajectoryPoints, setTrajectoryPoints] = useState<[number, number, number][]>([])

  useEffect(() => {
    if (!isRunning) return

    // Simulate vehicle movement (replace with real data from socket)
    const interval = setInterval(() => {
      setVehiclePos(prev => {
        const newX = prev[0] + Math.random() * 0.1 - 0.05
        const newZ = prev[2] + 0.05
        const newPos: [number, number, number] = [newX, 0.15, newZ]
        setTrajectoryPoints(prev => [...prev, newPos].slice(-100))
        return newPos
      })
      setVehicleRot(prev => prev + (Math.random() * 0.1 - 0.05))
    }, 50)

    return () => clearInterval(interval)
  }, [isRunning])

  return (
    <Canvas shadows className="w-full h-full">
      <PerspectiveCamera makeDefault position={[8, 6, 8]} fov={60} />
      <OrbitControls enableDamping dampingFactor={0.05} />
      
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <directionalLight
        position={[10, 10, 5]}
        intensity={1}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <pointLight position={[0, 5, 0]} intensity={0.5} color="#00ffff" />
      <pointLight position={[5, 2, 5]} intensity={0.3} color="#ff00ff" />
      
      {/* Environment */}
      <Grid
        args={[20, 20]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#00ffff"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#ff00ff"
        fadeDistance={30}
        fadeStrength={1}
        position={[0, -0.01, 0]}
      />
      
      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[50, 50]} />
        <meshStandardMaterial color="#0a0a0f" roughness={0.8} />
      </mesh>
      
      {/* Vehicle */}
      <Vehicle3D position={vehiclePos} rotation={vehicleRot} isActive={isRunning} />
      
      {/* Trajectory path */}
      {trajectoryPoints.length > 1 && <TrajectoryPath points={trajectoryPoints} />}
      
      {/* Obstacles */}
      <Obstacle position={[2, 0.5, 3]} radius={0.5} />
      <Obstacle position={[-1, 0.5, 5]} radius={0.6} />
      <Obstacle position={[1.5, 0.5, 7]} radius={0.4} />
      
      {/* Goal gate */}
      <GoalGate position={[0, 0, 10]} width={2.5} />
    </Canvas>
  )
}
