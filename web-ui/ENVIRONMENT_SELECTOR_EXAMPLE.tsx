/**
 * Environment Selector Component
 * 
 * Add this to your web UI to allow users to switch between
 * simple demonstration environment and research-grade environment.
 * 
 * Place in: web-ui/src/components/EnvironmentSelector.tsx
 */

'use client';

import { useState, useEffect } from 'react';

interface EnvironmentType {
  id: string;
  name: string;
  description: string;
}

interface EnvironmentSelectorProps {
  socket: any;
  onEnvChange?: (envType: string) => void;
}

export default function EnvironmentSelector({ socket, onEnvChange }: EnvironmentSelectorProps) {
  const [envTypes, setEnvTypes] = useState<EnvironmentType[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string>('simple');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!socket) return;

    // Request available environment types
    socket.emit('get_environment_types');

    // Listen for response
    socket.on('environment_types', (data: any) => {
      setEnvTypes(data.types);
      setSelectedEnv(data.default);
      setLoading(false);
    });

    return () => {
      socket.off('environment_types');
    };
  }, [socket]);

  const handleEnvChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newEnv = e.target.value;
    setSelectedEnv(newEnv);
    if (onEnvChange) {
      onEnvChange(newEnv);
    }
  };

  if (loading) {
    return (
      <div className="environment-selector loading">
        Loading environments...
      </div>
    );
  }

  return (
    <div className="environment-selector">
      <label htmlFor="env-type" className="env-label">
        Environment:
      </label>
      <select
        id="env-type"
        value={selectedEnv}
        onChange={handleEnvChange}
        className="env-select"
      >
        {envTypes.map((env) => (
          <option key={env.id} value={env.id}>
            {env.name}
          </option>
        ))}
      </select>
      
      <div className="env-description">
        {envTypes.find(e => e.id === selectedEnv)?.description}
      </div>

      <style jsx>{`
        .environment-selector {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          padding: 1rem;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          background: #f9fafb;
        }

        .env-label {
          font-weight: 600;
          font-size: 0.875rem;
          color: #374151;
        }

        .env-select {
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.375rem;
          font-size: 0.875rem;
          background: white;
          cursor: pointer;
        }

        .env-select:hover {
          border-color: #9ca3af;
        }

        .env-select:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .env-description {
          font-size: 0.75rem;
          color: #6b7280;
          line-height: 1.4;
        }

        .loading {
          color: #6b7280;
          padding: 1rem;
        }
      `}</style>
    </div>
  );
}


/**
 * USAGE EXAMPLE
 * 
 * In your main component (e.g., LiveDemo.tsx):
 */

/*
import EnvironmentSelector from './EnvironmentSelector';

function LiveDemo() {
  const [socket, setSocket] = useState<any>(null);
  const [currentEnvType, setCurrentEnvType] = useState('simple');

  const startSimulation = (model: string) => {
    if (socket) {
      socket.emit('start_simulation', {
        model: model,
        scenario: 'loose',
        max_steps: 500,
        env_type: currentEnvType  // <-- Pass selected environment
      });
    }
  };

  return (
    <div>
      <EnvironmentSelector 
        socket={socket}
        onEnvChange={(envType) => setCurrentEnvType(envType)}
      />
      
      <button onClick={() => startSimulation('sac_loose_2')}>
        Start Simulation
      </button>
    </div>
  );
}
*/


/**
 * ALTERNATIVE: Simple HTML/JavaScript Version
 * 
 * If not using React, add this to your HTML:
 */

/*
<!-- HTML -->
<div class="environment-selector">
  <label for="env-type">Environment:</label>
  <select id="env-type">
    <option value="simple">Simple (Demo)</option>
    <option value="research">Research-Grade</option>
  </select>
  <p class="env-description">
    <span data-env="simple">Original environment - fast, works with old models</span>
    <span data-env="research" style="display: none;">
      Advanced with GPS, IMU, and EKF - requires new models
    </span>
  </p>
</div>

<script>
// JavaScript
const envSelect = document.getElementById('env-type');
const descriptions = document.querySelectorAll('.env-description span');

envSelect.addEventListener('change', (e) => {
  const selected = e.target.value;
  
  // Update description
  descriptions.forEach(desc => {
    desc.style.display = desc.dataset.env === selected ? 'block' : 'none';
  });
});

// When starting simulation:
function startSimulation(modelName) {
  const envType = document.getElementById('env-type').value;
  
  socket.emit('start_simulation', {
    model: modelName,
    scenario: 'loose',
    max_steps: 500,
    env_type: envType  // <-- Include environment type
  });
}
</script>
*/


/**
 * IMPORTANT NOTES:
 * 
 * 1. Model Compatibility Warning:
 *    Consider adding a warning when users select research env with old models:
 * 
 *    if (envType === 'research' && isOldModel(modelName)) {
 *      alert('Warning: This model was trained on the simple environment. ' +
 *            'It may not work correctly with the research environment.');
 *    }
 * 
 * 2. Model Detection:
 *    You might want to store metadata about which environment each model was trained on.
 * 
 * 3. Visual Indicator:
 *    Show the current environment type in the simulation display:
 * 
 *    socket.on('simulation_metrics', (data) => {
 *      document.getElementById('env-indicator').textContent = 
 *        `Environment: ${data.env_type}`;
 *    });
 */
