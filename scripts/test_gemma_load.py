#!/usr/bin/env python3
"""Test loading the existing Gemma 2 9B model."""

import os
import sys
from pathlib import Path

def test_basic_imports():
    """Test if we can import basic libraries."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import h5py
        print("✅ h5py imported successfully") 
    except ImportError as e:
        print(f"❌ h5py import failed: {e}")
        return False
    
    return True

def test_model_files():
    """Test if we can access the Gemma model files."""
    print("\nTesting model file access...")
    
    model_path = Path("models/gemma2_9b")
    
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    print(f"✅ Model directory found: {model_path}")
    
    # Check key files
    required_files = [
        "config.json",
        "metadata.json", 
        "model_00000.weights.h5",
        "model_00001.weights.h5",
        "tokenizer.json"
    ]
    
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file}: {size_mb:.1f} MB")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    return True

def test_simple_computation():
    """Test basic computation without TensorFlow."""
    print("\nTesting basic computation...")
    
    try:
        import numpy as np
        
        # Simple matrix operations
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000) 
        c = np.dot(a, b)
        
        print(f"✅ NumPy computation successful: {c.shape}")
        return True
    except Exception as e:
        print(f"❌ Computation failed: {e}")
        return False

def analyze_model_structure():
    """Analyze the existing model structure."""
    print("\nAnalyzing model structure...")
    
    try:
        import json
        import h5py
        
        model_path = Path("models/gemma2_9b")
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        print("Model Configuration:")
        print(f"  Class: {config.get('class_name', 'Unknown')}")
        print(f"  Layers: {config['config'].get('num_layers', 'Unknown')}")
        print(f"  Hidden dim: {config['config'].get('hidden_dim', 'Unknown')}")
        print(f"  Vocab size: {config['config'].get('vocabulary_size', 'Unknown')}")
        
        # Check H5 files
        h5_files = list(model_path.glob("*.h5"))
        total_size = sum(f.stat().st_size for f in h5_files) / (1024**3)
        
        print(f"\nModel Files:")
        print(f"  H5 weight files: {len(h5_files)}")
        print(f"  Total size: {total_size:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GEMMA MODEL LOADING TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Files", test_model_files),
        ("Simple Computation", test_simple_computation),
        ("Model Structure", analyze_model_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()