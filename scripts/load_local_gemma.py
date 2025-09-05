#!/usr/bin/env python3
"""
Load and test Gemma 2 9B model from local directory using TensorFlow/Keras.
Measures text generation performance on M3 Pro.
"""

import os
import time
import keras
import keras_hub
import tensorflow as tf

def load_gemma_model(model_path):
    """Load Gemma model from local directory."""
    print(f"Loading Gemma 2 9B model from: {model_path}")
    
    # Load the CausalLM model
    try:
        model = keras_hub.models.GemmaCausalLM.from_preset(model_path)
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def test_text_generation(model, prompt="The future of AI is", max_length=100):
    """Test basic text generation with the model."""
    print(f"\nTesting text generation with prompt: '{prompt}'")
    
    try:
        # Generate text
        start_time = time.time()
        generated_text = model.generate(prompt, max_length=max_length)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        print(f"Generated text:\n{generated_text}")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        return generated_text, generation_time
    except Exception as e:
        print(f"✗ Error during text generation: {e}")
        return None, None

def measure_tokens_per_second(model, prompt="The future of artificial intelligence", num_runs=3):
    """Measure tokens per second performance."""
    print(f"\nMeasuring tokens per second performance ({num_runs} runs)")
    
    times = []
    token_counts = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        
        try:
            start_time = time.time()
            generated_text = model.generate(prompt, max_length=50)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(generated_text) // 4
            
            times.append(generation_time)
            token_counts.append(estimated_tokens)
            
            print(f"  Time: {generation_time:.2f}s, Est. tokens: {estimated_tokens}")
            
        except Exception as e:
            print(f"  ✗ Error in run {i+1}: {e}")
            continue
    
    if times:
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        print(f"\nPerformance Summary:")
        print(f"Average generation time: {avg_time:.2f} seconds")
        print(f"Average tokens generated: {avg_tokens:.0f}")
        print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")
        
        return tokens_per_second
    else:
        print("No successful runs to calculate performance")
        return None

def main():
    print("=" * 60)
    print("Gemma 2 9B Local Model Loading and Testing")
    print("=" * 60)
    
    # Check system info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Keras Hub version: {keras_hub.__version__}")
    
    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("No GPUs detected, using CPU")
    
    # Model path
    model_path = "models/gemma2_9b"
    
    if not os.path.exists(model_path):
        print(f"✗ Model directory not found: {model_path}")
        return
    
    # Load model
    model = load_gemma_model(model_path)
    if model is None:
        return
    
    # Test basic generation
    generated_text, gen_time = test_text_generation(model)
    
    if generated_text is not None:
        # Measure performance
        tokens_per_second = measure_tokens_per_second(model)
        
        print("\n" + "=" * 60)
        print("Testing Complete!")
        if tokens_per_second:
            print(f"Final Performance: {tokens_per_second:.2f} tokens/second on M3 Pro")
        print("=" * 60)
    
if __name__ == "__main__":
    main()