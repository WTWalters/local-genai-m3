#!/usr/bin/env python3
"""
Load and test Gemma 2 9B model from local directory using TensorFlow directly.
Simplified version that avoids keras-hub compatibility issues.
"""

import os
import time
import json
import h5py
import numpy as np

def check_model_files(model_path):
    """Check what model files are available."""
    print(f"Checking model files in: {model_path}")
    
    files = []
    if os.path.exists(model_path):
        for root, dirs, filenames in os.walk(model_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, model_path)
                files.append(rel_path)
    
    print(f"Found {len(files)} files:")
    for file in sorted(files):
        size = os.path.getsize(os.path.join(model_path, file))
        size_mb = size / (1024 * 1024)
        print(f"  {file} ({size_mb:.1f} MB)")
    
    return files

def load_model_config(model_path):
    """Load model configuration."""
    config_path = os.path.join(model_path, "config.json")
    metadata_path = os.path.join(model_path, "metadata.json")
    
    config = None
    metadata = None
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ Loaded config.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("✓ Loaded metadata.json")
    
    return config, metadata

def examine_h5_weights(model_path):
    """Examine the H5 weight files."""
    print("\nExamining H5 weight files:")
    
    weight_files = []
    for file in os.listdir(model_path):
        if file.endswith('.h5') and 'weights' in file:
            weight_files.append(file)
    
    total_params = 0
    for weight_file in sorted(weight_files):
        file_path = os.path.join(model_path, weight_file)
        print(f"\n{weight_file}:")
        
        try:
            with h5py.File(file_path, 'r') as f:
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape
                        dtype = obj.dtype
                        params = np.prod(shape)
                        nonlocal total_params
                        total_params += params
                        print(f"  {name}: {shape} {dtype} ({params:,} params)")
                    elif isinstance(obj, h5py.Group):
                        print(f"  {name}/ (group)")
                
                f.visititems(print_structure)
        except Exception as e:
            print(f"  Error reading {weight_file}: {e}")
    
    print(f"\nTotal parameters found: {total_params:,}")
    return total_params

def load_tokenizer_vocab(model_path):
    """Load tokenizer vocabulary."""
    vocab_path = os.path.join(model_path, "assets", "tokenizer", "vocabulary.spm")
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    
    vocab_size = 0
    
    if os.path.exists(vocab_path):
        size = os.path.getsize(vocab_path)
        print(f"✓ Found SentencePiece vocabulary: {size/1024:.1f} KB")
    
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, 'r') as f:
            tokenizer_data = json.load(f)
            if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                vocab_size = len(tokenizer_data['model']['vocab'])
                print(f"✓ Tokenizer vocabulary size: {vocab_size:,} tokens")
    
    return vocab_size

def estimate_performance(total_params, vocab_size):
    """Estimate performance characteristics for M3 Pro."""
    print(f"\nPerformance Estimation for M3 Pro:")
    print(f"Model parameters: {total_params/1e9:.1f}B")
    print(f"Vocabulary size: {vocab_size:,}")
    
    # M3 Pro specifications (approximate)
    m3_pro_memory_bandwidth = 150  # GB/s
    m3_pro_compute_tflops = 5.1   # TFLOPS (FP16)
    
    # Rough estimates
    model_size_gb = total_params * 2 / 1e9  # Assuming FP16
    print(f"Estimated model size: {model_size_gb:.1f} GB")
    
    # Very rough token generation speed estimate
    # This is a simplified calculation and actual performance will vary
    estimated_tokens_per_sec = min(
        m3_pro_memory_bandwidth / model_size_gb,  # Memory bound
        m3_pro_compute_tflops * 1e12 / (total_params * 4)  # Compute bound (rough)
    )
    
    print(f"Estimated tokens/sec: {estimated_tokens_per_sec:.1f} (very rough estimate)")
    print("\nNote: Actual performance depends on:")
    print("- Model architecture efficiency")
    print("- Memory optimization")
    print("- Batch size")
    print("- Sequence length")
    print("- Framework overhead")

def simple_benchmark():
    """Run a simple computational benchmark."""
    print(f"\nRunning simple M3 Pro benchmark:")
    
    # Simple matrix multiplication benchmark
    size = 2048
    iterations = 5
    
    print(f"Matrix multiplication {size}x{size}, {iterations} iterations...")
    
    times = []
    for i in range(iterations):
        a = np.random.randn(size, size).astype(np.float16)
        b = np.random.randn(size, size).astype(np.float16)
        
        start_time = time.time()
        c = np.dot(a, b)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        operations = 2 * size ** 3  # Approximate FLOPS for matrix multiply
        gflops = operations / elapsed / 1e9
        
        print(f"  Run {i+1}: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
    
    avg_time = sum(times) / len(times)
    avg_gflops = 2 * size ** 3 / avg_time / 1e9
    print(f"Average: {avg_time:.3f}s ({avg_gflops:.1f} GFLOPS)")

def main():
    print("=" * 70)
    print("Gemma 2 9B Local Model Analysis")
    print("=" * 70)
    
    # Check Python/system info
    print(f"Python version: {os.sys.version.split()[0]}")
    print(f"Platform: {os.sys.platform}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("No GPUs detected, using CPU")
    except ImportError:
        print("TensorFlow not available")
    
    # Model path
    model_path = "models/gemma2_9b"
    
    if not os.path.exists(model_path):
        print(f"✗ Model directory not found: {model_path}")
        return
    
    # Check model files
    files = check_model_files(model_path)
    
    # Load configurations
    config, metadata = load_model_config(model_path)
    
    if metadata:
        print(f"\nModel Info:")
        print(f"Parameters: {metadata.get('parameter_count', 'Unknown'):,}")
        print(f"Keras version: {metadata.get('keras_version', 'Unknown')}")
        print(f"Date saved: {metadata.get('date_saved', 'Unknown')}")
    
    # Examine weights
    total_params = examine_h5_weights(model_path)
    
    # Load tokenizer info
    vocab_size = load_tokenizer_vocab(model_path)
    
    # Performance estimation
    if total_params > 0:
        estimate_performance(total_params, vocab_size)
    
    # Simple benchmark
    simple_benchmark()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nTo actually load and run this model, you would need:")
    print("1. A compatible keras-hub environment")
    print("2. Sufficient memory (16+ GB recommended)")
    print("3. Proper model loading code using keras_hub.models.GemmaCausalLM")

if __name__ == "__main__":
    main()