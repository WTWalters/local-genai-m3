#!/usr/bin/env python3
"""Setup Gemma model - simplified version for initial testing."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path

def test_basic_model():
    """Test basic TensorFlow model loading."""
    
    print("Testing basic TensorFlow model operations...")
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Create a simple model to test
    print("\nCreating test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print(f"Model created successfully")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test inference
    import numpy as np
    test_input = np.random.randn(1, 100).astype(np.float32)
    
    with tf.device('/GPU:0'):
        output = model(test_input)
    
    print(f"\nTest inference successful")
    print(f"Output shape: {output.shape}")
    
    return model

def download_gemma_info():
    """Information about downloading Gemma models."""
    
    print("\n" + "="*60)
    print("GEMMA MODEL SETUP")
    print("="*60)
    
    print("""
To use Gemma models, you need to:

1. Create a Kaggle account at https://www.kaggle.com

2. Accept the Gemma license agreement:
   - Visit: https://www.kaggle.com/models/google/gemma-2
   - Click 'Request Access'
   - Accept the terms

3. Get your Kaggle API credentials:
   - Go to: https://www.kaggle.com/settings
   - Scroll to 'API' section
   - Click 'Create New Token'
   - Save the downloaded kaggle.json file

4. Set up credentials:
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

Once setup, we can download Gemma models programmatically.
""")

if __name__ == "__main__":
    # Test basic model operations
    model = test_basic_model()
    
    # Show Gemma setup info
    download_gemma_info()