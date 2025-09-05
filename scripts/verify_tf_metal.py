#!/usr/bin/env python3
"""Verify TensorFlow Metal support on M3 Pro."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF verbosity

import tensorflow as tf
import platform
import psutil
import sys

def verify_tensorflow_metal():
    """Comprehensive TensorFlow Metal verification."""
    print("=" * 60)
    print("TENSORFLOW METAL VERIFICATION - M3 Pro")
    print("=" * 60)
    
    # System info
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    # TensorFlow info
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check for Metal device
    devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    print(f"\nAvailable devices:")
    for device in devices:
        print(f"  - {device}")
    
    if gpu_devices:
        print(f"\n✅ Metal GPU detected: {gpu_devices[0]}")
        
        # Configure memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
            print("✅ Memory growth enabled")
        except:
            pass
        
        # Test Metal with matrix operations
        print("\nTesting Metal performance...")
        with tf.device('/GPU:0'):
            # Test with large matrices
            a = tf.random.normal([4096, 4096])
            b = tf.random.normal([4096, 4096])
            
            # Warm up
            _ = tf.matmul(a, b)
            
            # Benchmark
            import time
            start = time.time()
            for _ in range(10):
                c = tf.matmul(a, b)
            tf.debugging.assert_all_finite(c, "Matrix multiplication")
            elapsed = time.time() - start
            
            print(f"✅ Metal computation successful")
            print(f"   4096x4096 matrix multiply (10 iterations): {elapsed:.2f}s")
            print(f"   TFLOPS: {(10 * 2 * 4096**3) / (elapsed * 1e12):.2f}")
    else:
        print("❌ No Metal GPU detected")
        print("   Check tensorflow-metal installation")
    
    # Memory info
    print(f"\nMemory Configuration:")
    if gpu_devices:
        try:
            # Set memory limit for M3 Pro (leave 6GB for system)
            tf.config.set_logical_device_configuration(
                gpu_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=30000)]
            )
            print(f"✅ GPU memory limit set to 30GB")
        except:
            print("   Using automatic memory management")
    
    print("=" * 60)

if __name__ == "__main__":
    verify_tensorflow_metal()