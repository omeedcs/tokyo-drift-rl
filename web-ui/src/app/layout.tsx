import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Deep RL for Autonomous Drifting',
  description: 'Deep Reinforcement Learning for Autonomous Vehicle Drifting - Interactive Research Platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
