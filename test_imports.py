#!/usr/bin/env python3
"""Quick test to verify all simulator components can be imported."""

import sys

def test_imports():
    """Test that all simulator modules can be imported."""
    print("🧪 Testing simulator imports...\n")
    
    tests = [
        ("Core dependencies", ["numpy", "torch", "matplotlib", "scipy"]),
        ("Simulator vehicle", ["src.simulator.vehicle"]),
        ("Simulator sensors", ["src.simulator.sensors"]),
        ("Simulator environment", ["src.simulator.environment"]),
        ("Simulator controller", ["src.simulator.controller"]),
        ("Simulator visualization", ["src.simulator.visualization"]),
    ]
    
    failed = []
    
    for test_name, modules in tests:
        try:
            for module in modules:
                __import__(module)
            print(f"✅ {test_name}")
        except ImportError as e:
            print(f"❌ {test_name}: {e}")
            failed.append(test_name)
    
    print()
    
    if failed:
        print(f"❌ {len(failed)} test(s) failed")
        print("Run './quick_setup.sh' to install dependencies")
        return False
    else:
        print("✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
