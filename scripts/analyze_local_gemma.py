#!/usr/bin/env python3
"""
Analyze Gemma 2 9B model files and estimate performance without TensorFlow dependencies.
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
    total_size = 0
    if os.path.exists(model_path):
        for root, dirs, filenames in os.walk(model_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, model_path)
                file_size = os.path.getsize(full_path)
                files.append((rel_path, file_size))
                total_size += file_size
    
    print(f"Found {len(files)} files (Total: {total_size/1024/1024/1024:.2f} GB):")
    for file, size in sorted(files):
        size_mb = size / (1024 * 1024)
        print(f"  {file:<30} ({size_mb:.1f} MB)")
    
    return files, total_size

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

def analyze_model_architecture(config):
    """Analyze model architecture from config."""
    if not config or 'config' not in config:
        print("No architecture config available")
        return
    
    arch = config['config']
    print(f"\nModel Architecture:")
    print(f"  Layers: {arch.get('num_layers', 'Unknown')}")
    print(f"  Hidden dim: {arch.get('hidden_dim', 'Unknown'):,}")
    print(f"  Intermediate dim: {arch.get('intermediate_dim', 'Unknown'):,}")
    print(f"  Query heads: {arch.get('num_query_heads', 'Unknown')}")
    print(f"  Key-value heads: {arch.get('num_key_value_heads', 'Unknown')}")
    print(f"  Head dim: {arch.get('head_dim', 'Unknown')}")
    print(f"  Vocabulary size: {arch.get('vocabulary_size', 'Unknown'):,}")
    print(f"  Sliding window: {arch.get('sliding_window_size', 'Unknown')}")

def examine_h5_weights(model_path):
    """Examine the H5 weight files."""
    print("\nExamining H5 weight files:")
    
    weight_files = []
    for file in os.listdir(model_path):
        if file.endswith('.h5') and 'weights' in file:
            weight_files.append(file)
    
    total_params = 0
    layer_info = {}
    
    for weight_file in sorted(weight_files):
        file_path = os.path.join(model_path, weight_file)
        file_size = os.path.getsize(file_path)
        print(f"\n{weight_file} ({file_size/1024/1024:.1f} MB):")
        
        try:
            with h5py.File(file_path, 'r') as f:
                def analyze_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape
                        dtype = obj.dtype
                        params = np.prod(shape)
                        nonlocal total_params
                        total_params += params
                        
                        # Extract layer type from name
                        layer_type = name.split('/')[0] if '/' in name else 'unknown'
                        if layer_type not in layer_info:
                            layer_info[layer_type] = {'params': 0, 'tensors': 0}
                        layer_info[layer_type]['params'] += params
                        layer_info[layer_type]['tensors'] += 1
                        
                        print(f"    {name:<50} {str(shape):<20} {dtype} ({params:,} params)")
                
                f.visititems(analyze_structure)
        except Exception as e:
            print(f"    Error reading {weight_file}: {e}")
    
    print(f"\nParameter Summary by Layer Type:")
    for layer_type, info in sorted(layer_info.items()):
        params_m = info['params'] / 1e6
        print(f"  {layer_type:<20} {info['tensors']:>3} tensors, {params_m:>8.1f}M params")
    
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    return total_params

def load_tokenizer_info(model_path):
    """Load tokenizer information."""
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
            
            # Show some tokenizer details
            if 'model' in tokenizer_data:
                model_type = tokenizer_data['model'].get('type', 'Unknown')
                print(f"✓ Tokenizer type: {model_type}")
    
    return vocab_size

def estimate_m3_pro_performance(total_params, model_size_gb):
    """Estimate performance on M3 Pro."""
    print(f"\n" + "="*60)
    print(f"M3 Pro Performance Estimation")
    print(f"="*60)
    
    # M3 Pro specifications
    m3_pro_specs = {
        'memory_bandwidth': 150,  # GB/s unified memory
        'cpu_cores': 12,          # 6P + 6E
        'gpu_cores': 18,          # GPU cores
        'neural_engine': 16,      # TOPS
        'max_memory': 36,         # GB (for higher configs)
    }
    
    print(f"Hardware Specifications:")
    for key, value in m3_pro_specs.items():
        unit = 'GB/s' if 'bandwidth' in key else 'TOPS' if 'neural' in key else 'GB' if 'memory' in key else ''
        print(f"  {key.replace('_', ' ').title()}: {value} {unit}")
    
    print(f"\nModel Requirements:")
    print(f"  Model size: {model_size_gb:.1f} GB")
    print(f"  Parameters: {total_params/1e9:.1f}B")
    
    # Memory analysis
    memory_utilization = (model_size_gb / m3_pro_specs['max_memory']) * 100
    print(f"  Memory utilization: {memory_utilization:.1f}%")
    
    if memory_utilization > 90:
        print("  ⚠️  High memory usage - may cause swapping")
    elif memory_utilization > 70:
        print("  ⚠️  Moderate memory usage")
    else:
        print("  ✅ Acceptable memory usage")
    
    # Rough performance estimates
    print(f"\nPerformance Estimates (rough):")
    
    # Memory bandwidth limited estimate
    memory_bound_tokens_sec = m3_pro_specs['memory_bandwidth'] / model_size_gb
    print(f"  Memory-bound tokens/sec: ~{memory_bound_tokens_sec:.1f}")
    
    # Neural Engine estimate (very rough)
    ne_tokens_sec = m3_pro_specs['neural_engine'] * 0.1  # Very conservative estimate
    print(f"  Neural Engine tokens/sec: ~{ne_tokens_sec:.1f} (if optimized)")
    
    # Conservative estimate
    realistic_estimate = min(memory_bound_tokens_sec * 0.3, 5.0)  # Conservative factor
    print(f"  Realistic estimate: ~{realistic_estimate:.1f} tokens/sec")
    
    print(f"\nFactors affecting actual performance:")
    print(f"  • Framework efficiency (Keras, MLX, etc.)")
    print(f"  • Quantization (FP16, INT8, etc.)")
    print(f"  • Batch size and sequence length")
    print(f"  • Memory optimization techniques")
    print(f"  • Model-specific optimizations")

def run_simple_benchmark():
    """Run a simple numpy benchmark."""
    print(f"\n" + "="*60)
    print(f"Simple Computational Benchmark")
    print(f"="*60)
    
    # Matrix multiplication benchmark
    sizes = [1024, 2048]
    dtypes = [np.float32, np.float16]
    
    for size in sizes:
        for dtype in dtypes:
            print(f"\nMatrix multiply {size}x{size} ({dtype.__name__}):")
            
            times = []
            for i in range(3):
                a = np.random.randn(size, size).astype(dtype)
                b = np.random.randn(size, size).astype(dtype)
                
                start_time = time.time()
                c = np.dot(a, b)
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                
                operations = 2 * size ** 3
                gflops = operations / elapsed / 1e9
                
                print(f"  Run {i+1}: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
            
            avg_time = sum(times) / len(times)
            avg_gflops = 2 * size ** 3 / avg_time / 1e9
            print(f"  Average: {avg_time:.3f}s ({avg_gflops:.1f} GFLOPS)")

def main():
    print("=" * 70)
    print("Gemma 2 9B Local Model Analysis")
    print("=" * 70)
    
    # System info
    print(f"Python version: {os.sys.version.split()[0]}")
    print(f"Platform: {os.sys.platform}")
    print(f"NumPy version: {np.__version__}")
    
    # Model path
    model_path = "models/gemma2_9b"
    
    if not os.path.exists(model_path):
        print(f"✗ Model directory not found: {model_path}")
        return
    
    print(f"\n" + "="*70)
    
    # Analyze model files
    files, total_size = check_model_files(model_path)
    
    # Load configurations
    config, metadata = load_model_config(model_path)
    
    if metadata:
        print(f"\nModel Metadata:")
        print(f"  Parameters: {metadata.get('parameter_count', 'Unknown'):,}")
        print(f"  Keras version: {metadata.get('keras_version', 'Unknown')}")
        print(f"  KerasHub version: {metadata.get('keras_hub_version', 'Unknown')}")
        print(f"  Date saved: {metadata.get('date_saved', 'Unknown')}")
        print(f"  Tasks: {', '.join(metadata.get('tasks', ['Unknown']))}")
    
    # Analyze architecture
    if config:
        analyze_model_architecture(config)
    
    # Examine weights
    total_params = examine_h5_weights(model_path)
    
    # Tokenizer info
    vocab_size = load_tokenizer_info(model_path)
    
    # Performance estimation
    model_size_gb = total_size / (1024**3)
    estimate_m3_pro_performance(total_params, model_size_gb)
    
    # Simple benchmark
    run_simple_benchmark()
    
    print(f"\n" + "="*70)
    print("Analysis Complete!")
    print(f"="*70)

if __name__ == "__main__":
    main()