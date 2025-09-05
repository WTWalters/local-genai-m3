# TensorFlow/Gemma Implementation Guide: Local Gen AI on MacBook M3 Pro

## Overview
This guide implements a local generative AI system using TensorFlow (leveraging your existing expertise) and Google's Gemma 2 model, optimized for your M3 Pro with 36GB RAM.

## Prerequisites

### System Requirements
- ✅ MacBook Pro M3 Pro with 36GB RAM (your machine)
- macOS Sequoia 15.6.1 or later (✅ you have 15.6.1)
- 50GB+ free disk space
- Xcode Command Line Tools

### Initial Setup
```bash
# Install Xcode Command Line Tools (if needed)
xcode-select --install

# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install cmake git git-lfs wget protobuf
```

## Phase 1: TensorFlow Environment Setup

### Step 1.1: Create Project Structure
```bash
cd /Users/whitneywalters/AIProgramming/tuning
mkdir -p {models,data,scripts,configs,checkpoints,logs,chroma_db,business_docs}

# Create project README
cat > README.md << 'EOF'
# Local GenAI System - TensorFlow/Gemma Implementation
Optimized for M3 Pro with TensorFlow Metal acceleration
EOF
```

### Step 1.2: Create TensorFlow Environment
```bash
# Create conda environment
conda create -n tf_gemma python=3.11 -y
conda activate tf_gemma

# Create environment file
cat > environment.yml << 'EOF'
name: tf_gemma
channels:
  - conda-forge
  - apple
dependencies:
  - python=3.11
  - pip
  - numpy<2.0
  - scipy
  - jupyter
  - pip:
    - tensorflow==2.15.0
    - tensorflow-metal==1.1.0
    - tensorflow-datasets
    - keras-nlp==0.14.0
    - keras==3.0.0
    - kagglehub
    - chromadb==0.5.5
    - docling==2.48.0
    - sentence-transformers==3.0.1
    - langchain==0.2.14
    - langchain-community==0.2.12
    - pandas
    - tqdm
    - python-dotenv
    - protobuf==3.20.3
EOF

conda env update -f environment.yml

# Install TensorFlow Metal plugin
pip install tensorflow-metal==1.1.0
```

### Step 1.3: Verify TensorFlow Metal Support
Create `scripts/verify_tf_metal.py`:
```python
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
```

### Step 1.4: Environment Configuration
Create `.env` file:
```bash
cat > .env << 'EOF'
# TensorFlow Metal Configuration for M3 Pro
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export METAL_DEVICE_SUPPORT_ENABLED=1

# Memory Management (36GB RAM optimized)
export TF_GPU_MEMORY_LIMIT=30000
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Gemma Model Configuration
export GEMMA_MODEL=gemma2_instruct_9b_en
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_key

# Project Paths
export PROJECT_ROOT=/Users/whitneywalters/AIProgramming/tuning
export MODEL_PATH=$PROJECT_ROOT/models
export DATA_PATH=$PROJECT_ROOT/data
export CHECKPOINT_PATH=$PROJECT_ROOT/checkpoints

# Training Configuration
export BATCH_SIZE=2
export LEARNING_RATE=2e-5
export MAX_LENGTH=2048
EOF

source .env
```

## Phase 2: Gemma Model Setup

### Step 2.1: Kaggle Authentication for Gemma
```bash
# Create Kaggle credentials (you'll need a Kaggle account)
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2.2: Download and Setup Gemma
Create `scripts/setup_gemma.py`:
```python
#!/usr/bin/env python3
"""Download and setup Gemma 2 model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_nlp
from pathlib import Path
import kagglehub

def download_gemma():
    """Download Gemma 2 model from Kaggle."""
    
    print("Downloading Gemma 2 9B model...")
    print("Note: You need to accept the license at https://www.kaggle.com/models/google/gemma-2")
    
    # Download model weights via kagglehub
    model_path = kagglehub.model_download(
        "google/gemma-2/keras/gemma2_instruct_9b_en",
        path=str(Path(os.environ['MODEL_PATH']) / "gemma2_9b")
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

def load_gemma_model(preset="gemma2_instruct_2b_en"):
    """Load Gemma model with TensorFlow."""
    
    print(f"Loading Gemma model: {preset}")
    
    # Configure GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Load model using Keras NLP
    model = keras_nlp.models.GemmaCausalLM.from_preset(
        preset,
        dtype="mixed_float16"  # Use mixed precision for efficiency
    )
    
    print(f"Model loaded successfully")
    print(f"Parameters: {model.count_params():,}")
    
    return model

def test_gemma_inference():
    """Test Gemma inference."""
    
    # Start with smaller model for testing
    model = load_gemma_model("gemma2_instruct_2b_en")
    
    # Test generation
    prompt = "Explain the benefits of local AI deployment for businesses:"
    
    print(f"\nPrompt: {prompt}")
    print("Generating response...")
    
    response = model.generate(
        prompt,
        max_length=256,
        temperature=0.7
    )
    
    print(f"Response: {response}")

if __name__ == "__main__":
    # Download if needed
    # model_path = download_gemma()
    
    # Test with smaller model first
    test_gemma_inference()
```

### Step 2.3: Convert Gemma to GGUF
Create `scripts/convert_gemma_gguf.py`:
```python
#!/usr/bin/env python3
"""Convert Gemma model to GGUF format for llama.cpp inference."""

import os
import subprocess
from pathlib import Path
import tensorflow as tf
import json

def export_gemma_to_safetensors(model_path, output_path):
    """Export Gemma to safetensors format."""
    
    import keras_nlp
    from safetensors.tensorflow import save_file
    
    print("Loading Gemma model...")
    model = keras_nlp.models.GemmaCausalLM.from_preset(
        "gemma2_instruct_2b_en",  # Start with 2B for testing
        dtype="float32"
    )
    
    # Extract weights
    weights = {}
    for var in model.weights:
        name = var.name.replace(":", "_").replace("/", ".")
        weights[name] = var.numpy()
    
    # Save as safetensors
    output_file = Path(output_path) / "model.safetensors"
    save_file(weights, output_file)
    print(f"Saved to {output_file}")
    
    return output_file

def convert_to_gguf(model_path, output_path, quantization="Q5_K_M"):
    """Convert model to GGUF using llama.cpp."""
    
    llama_cpp_path = Path(os.environ['PROJECT_ROOT']) / "llama.cpp"
    
    # Build llama.cpp if not already built
    if not llama_cpp_path.exists():
        print("Building llama.cpp with Metal support...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_cpp_path)
        ], check=True)
        
        subprocess.run(
            ["make", "LLAMA_METAL=1", "-j8"],
            cwd=llama_cpp_path,
            check=True
        )
    
    # Convert to GGUF
    convert_script = llama_cpp_path / "convert-gemma-to-gguf.py"
    
    cmd = [
        "python", str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization.lower()
    ]
    
    print(f"Converting to GGUF ({quantization})...")
    subprocess.run(cmd, check=True)
    
    return output_path

if __name__ == "__main__":
    model_path = Path(os.environ['MODEL_PATH']) / "gemma2_2b"
    output_path = Path(os.environ['MODEL_PATH']) / "gemma2_2b.Q5_K_M.gguf"
    
    # For production, use the conversion pipeline
    # For now, we'll use pre-converted models when available
    print("Note: Gemma GGUF conversion requires additional setup")
    print("Consider using pre-converted GGUF models from Hugging Face")
```

## Phase 3: RAG Pipeline with TensorFlow

### Step 3.1: Document Processing
Create `scripts/process_documents_tf.py`:
```python
#!/usr/bin/env python3
"""Process documents using Docling for RAG pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
import tensorflow as tf
from tqdm import tqdm

class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self, corpus_path: str = "./business_docs"):
        self.corpus_path = Path(corpus_path)
        self.converter = DocumentConverter()
        
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document."""
        try:
            result = self.converter.convert(str(file_path))
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                "source": file_path.name,
                "path": str(file_path),
                "format": file_path.suffix,
            }
            
            return {
                "content": markdown_content,
                "metadata": metadata,
                "status": "success"
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {
                "content": None,
                "metadata": {"source": file_path.name},
                "status": "error",
                "error": str(e)
            }
    
    def chunk_with_sliding_window(self, text: str, window_size: int = 512, stride: int = 256):
        """Chunk text using sliding window (Gemma-optimized)."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), stride):
            chunk = ' '.join(words[i:i + window_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
            
            if i + window_size >= len(words):
                break
        
        return chunks
    
    def process_corpus(self) -> List[Dict[str, Any]]:
        """Process all documents in corpus."""
        all_chunks = []
        
        # Get all documents
        doc_files = list(self.corpus_path.glob("**/*"))
        doc_files = [f for f in doc_files if f.suffix in [".pdf", ".docx", ".txt", ".md"]]
        
        print(f"Found {len(doc_files)} documents to process")
        
        for file_path in tqdm(doc_files, desc="Processing documents"):
            doc_data = self.process_document(file_path)
            
            if doc_data["status"] == "success" and doc_data["content"]:
                chunks = self.chunk_with_sliding_window(doc_data["content"])
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "content": chunk,
                        "metadata": {
                            **doc_data["metadata"],
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    })
        
        print(f"Created {len(all_chunks)} chunks from {len(doc_files)} documents")
        
        # Save chunks
        output_path = Path("data/processed_chunks.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(all_chunks, f, indent=2)
        
        return all_chunks

if __name__ == "__main__":
    # Add sample documents to process
    sample_dir = Path("business_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a sample document if none exist
    sample_file = sample_dir / "sample_business_doc.md"
    if not sample_file.exists():
        sample_file.write_text("""
# Q3 2024 Business Report

## Executive Summary
Revenue increased by 15% YoY to $45.2M, driven by strong performance in Project Phoenix.

## Key Metrics
- Customer acquisition: +22%
- Market expansion: 3 new regions
- Operating efficiency: 8% improvement

## Strategic Initiatives
1. Digital transformation program
2. AI integration for customer service
3. Supply chain optimization
        """)
    
    processor = DocumentProcessor()
    chunks = processor.process_corpus()
```

### Step 3.2: Vector Database with TensorFlow Embeddings
Create `scripts/setup_vectordb_tf.py`:
```python
#!/usr/bin/env python3
"""Setup ChromaDB with TensorFlow-compatible embeddings."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class VectorDBManager:
    """Manage ChromaDB with TensorFlow optimization."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "gemma_knowledge"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="mps"  # Use Metal Performance Shaders
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Gemma-optimized knowledge base"}
            )
            print(f"Created new collection: {self.collection_name}")
        except:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
    
    @tf.function
    def preprocess_batch(self, texts):
        """TensorFlow preprocessing for batch efficiency."""
        # This would integrate with TF preprocessing if needed
        return texts
    
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        """Add documents with TF-optimized batching."""
        
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i:06d}" for i in range(len(chunks))]
        
        # Process in optimized batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector DB"):
            batch_texts = texts[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Add prefix for nomic model
            batch_texts_with_prefix = [f"search_document: {text}" for text in batch_texts]
            
            # Generate embeddings
            batch_embeddings = self.embedding_model.encode(
                batch_texts_with_prefix,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size
            )
            
            # Add to collection
            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        
        print(f"Added {len(chunks)} chunks to vector database")
        print(f"Total documents: {self.collection.count()}")
    
    def search_tf(self, query: str, n_results: int = 5):
        """TensorFlow-optimized search."""
        
        # Add query prefix
        query_with_prefix = f"search_query: {query}"
        
        # Generate embedding
        with tf.device('/GPU:0'):
            query_embedding = self.embedding_model.encode(
                query_with_prefix,
                convert_to_numpy=True
            )
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results

if __name__ == "__main__":
    # Load processed chunks
    chunks_path = Path("data/processed_chunks.json")
    
    if not chunks_path.exists():
        print("No processed chunks found. Run process_documents_tf.py first.")
        exit(1)
    
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    # Setup vector database
    db_manager = VectorDBManager()
    db_manager.add_documents(chunks, batch_size=64)  # Optimized for M3 Pro
    
    # Test search
    test_query = "What are the Q3 revenue figures?"
    results = db_manager.search_tf(test_query, n_results=3)
    
    print("\nTest Search Results:")
    print(f"Query: {test_query}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"   Content: {result['content'][:200]}...")
```

## Phase 4: Fine-Tuning with Keras

### Step 4.1: Prepare Fine-Tuning Dataset
Create `scripts/prepare_finetuning_tf.py`:
```python
#!/usr/bin/env python3
"""Prepare fine-tuning dataset for Gemma with TensorFlow."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflow as tf
from pathlib import Path
from typing import List, Dict

class GemmaDatasetBuilder:
    """Build instruction-tuning dataset for Gemma."""
    
    def __init__(self, output_path: str = "data/gemma_finetune.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True)
        
        # Gemma 2 instruction format
        self.instruction_template = """<start_of_turn>user
{instruction}

Context: {context}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>"""
    
    def create_business_examples(self) -> List[Dict[str, str]]:
        """Create business-specific training examples."""
        
        examples = [
            {
                "instruction": "Summarize the quarterly performance",
                "context": "Q3 2024 showed revenue of $45.2M with 15% YoY growth. Project Phoenix contributed $5M. Operating efficiency improved by 8%.",
                "response": "Q3 2024 delivered strong results with $45.2M in revenue, representing 15% year-over-year growth. The successful Project Phoenix launch added $5M to revenues, while operational improvements yielded an 8% efficiency gain."
            },
            {
                "instruction": "Identify key risks from the assessment",
                "context": "Risk Assessment: 1) Cybersecurity (High/High), 2) Supply chain (High/Medium), 3) Regulatory (Medium/High), 4) Talent retention (Medium/Medium)",
                "response": "The assessment identifies cybersecurity as the top priority risk with high impact and probability. Supply chain disruptions pose high impact but medium probability. Regulatory changes are highly probable with medium impact. Talent retention represents a balanced medium-level risk."
            },
            {
                "instruction": "What are our strategic priorities?",
                "context": "Strategic initiatives include: Digital transformation program targeting 20% efficiency gains, AI integration for customer service automation, and supply chain diversification across 3 new regions.",
                "response": "Our strategic priorities focus on three key areas: First, digital transformation aimed at achieving 20% efficiency improvements. Second, implementing AI-powered customer service automation. Third, diversifying our supply chain by expanding into three new geographic regions."
            }
        ]
        
        return examples
    
    def format_for_gemma(self, examples: List[Dict[str, str]]) -> tf.data.Dataset:
        """Format examples as TensorFlow dataset."""
        
        formatted = []
        for ex in examples:
            text = self.instruction_template.format(
                instruction=ex["instruction"],
                context=ex["context"],
                response=ex["response"]
            )
            
            formatted.append({
                "text": text,
                "instruction": ex["instruction"],
                "response": ex["response"]
            })
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: formatted,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "instruction": tf.TensorSpec(shape=(), dtype=tf.string),
                "response": tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
        
        return dataset, formatted
    
    def save_dataset(self, examples: List[Dict]):
        """Save dataset to JSON."""
        with open(self.output_path, "w") as f:
            json.dump(examples, f, indent=2)
        
        print(f"Saved {len(examples)} examples to {self.output_path}")
    
    def build_dataset(self):
        """Build complete dataset."""
        examples = self.create_business_examples()
        dataset, formatted = self.format_for_gemma(examples)
        self.save_dataset(formatted)
        
        return dataset, formatted

if __name__ == "__main__":
    builder = GemmaDatasetBuilder()
    dataset, examples = builder.build_dataset()
    
    # Preview dataset
    print("\nDataset preview:")
    for i, example in enumerate(dataset.take(2)):
        print(f"\nExample {i+1}:")
        print(example['text'].numpy().decode('utf-8')[:200] + "...")
```

### Step 4.2: LoRA Fine-Tuning with Keras
Create `scripts/finetune_gemma_lora.py`:
```python
#!/usr/bin/env python3
"""Fine-tune Gemma with LoRA using Keras/TensorFlow."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_nlp
from pathlib import Path
import json

class GemmaLoRATrainer:
    """Fine-tune Gemma with LoRA on TensorFlow."""
    
    def __init__(self, model_preset: str = "gemma2_instruct_2b_en"):
        self.model_preset = model_preset
        self.checkpoint_dir = Path(os.environ['CHECKPOINT_PATH']) / "gemma_lora"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure GPU
        self.setup_gpu()
    
    def setup_gpu(self):
        """Configure Metal GPU for training."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpus[0], True)
                
                # Set memory limit (30GB for M3 Pro)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=30000)]
                )
                print(f"✅ GPU configured: {gpus[0]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    def create_lora_model(self):
        """Create Gemma model with LoRA adapters."""
        
        print(f"Loading base model: {self.model_preset}")
        
        # Load base model
        base_model = keras_nlp.models.GemmaCausalLM.from_preset(
            self.model_preset,
            dtype="mixed_float16"
        )
        
        # Enable LoRA on attention layers
        base_model.backbone.enable_lora(rank=16)
        
        print(f"Model parameters: {base_model.count_params():,}")
        print(f"Trainable parameters: {sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights):,}")
        
        return base_model
    
    def prepare_dataset(self, dataset_path: str):
        """Prepare dataset for training."""
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            examples = json.load(f)
        
        # Create TF dataset
        texts = [ex['text'] for ex in examples]
        
        dataset = tf.data.Dataset.from_tensor_slices(texts)
        dataset = dataset.batch(1)  # Small batch for M3 Pro
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self, dataset_path: str, epochs: int = 3):
        """Run LoRA fine-tuning."""
        
        # Create model
        model = self.create_lora_model()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset_path)
        
        # Configure optimizer with lower learning rate for LoRA
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=5e-5,
            weight_decay=0.01
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "checkpoint-{epoch:02d}"),
                save_weights_only=True,
                save_freq='epoch'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Train
        print("\nStarting LoRA fine-tuning...")
        print(f"Epochs: {epochs}")
        print(f"Dataset size: {len(list(train_dataset))}")
        
        history = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model"
        model.save_weights(str(final_path))
        print(f"\nModel saved to: {final_path}")
        
        return model, history
    
    def test_generation(self, model):
        """Test the fine-tuned model."""
        
        test_prompts = [
            "Summarize our Q3 performance:",
            "What are the main strategic risks?",
            "Explain our digital transformation strategy:"
        ]
        
        print("\n" + "="*60)
        print("Testing Fine-tuned Model")
        print("="*60)
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            response = model.generate(
                prompt,
                max_length=128,
                temperature=0.7
            )
            
            print(f"Response: {response}")
            print("-"*40)

if __name__ == "__main__":
    dataset_path = "data/gemma_finetune.json"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run prepare_finetuning_tf.py first")
        exit(1)
    
    # Initialize trainer
    trainer = GemmaLoRATrainer(
        model_preset="gemma2_instruct_2b_en"  # Start with 2B for testing
    )
    
    # Train
    model, history = trainer.train(dataset_path, epochs=3)
    
    # Test
    trainer.test_generation(model)
```

## Phase 5: Inference Pipeline

### Step 5.1: Unified RAG + Gemma Inference
Create `scripts/inference_pipeline_tf.py`:
```python
#!/usr/bin/env python3
"""Complete inference pipeline with RAG and fine-tuned Gemma."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_nlp
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any

class GemmaRAGPipeline:
    """RAG-enhanced Gemma inference pipeline."""
    
    def __init__(
        self,
        model_preset: str = "gemma2_instruct_2b_en",
        checkpoint_path: str = None,
        db_path: str = "./chroma_db",
        collection_name: str = "gemma_knowledge"
    ):
        # Setup GPU
        self.setup_gpu()
        
        # Load Gemma model
        print(f"Loading Gemma model: {model_preset}")
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(
            model_preset,
            dtype="mixed_float16"
        )
        
        # Load LoRA checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading LoRA weights from: {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="mps"
        )
        
        # Connect to ChromaDB
        print("Connecting to vector database...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        # Gemma prompt template
        self.prompt_template = """<start_of_turn>user
Use the following context to answer the question accurately.

Context:
{context}

Question: {question}<end_of_turn>
<start_of_turn>model"""
    
    def setup_gpu(self):
        """Configure Metal GPU."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    
    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from vector DB."""
        
        # Add query prefix for nomic
        query_with_prefix = f"search_query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Combine results
        contexts = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            source = metadata.get('source', 'Unknown')
            contexts.append(f"[Source: {source}]\n{doc}")
        
        return "\n---\n".join(contexts)
    
    @tf.function
    def generate_response_tf(self, input_ids, max_length=512):
        """TensorFlow-optimized generation."""
        # This would be the actual TF generation logic
        # For now, we use the Keras NLP generate method
        return input_ids
    
    def generate_response(self, query: str, use_rag: bool = True) -> str:
        """Generate response using RAG-enhanced Gemma."""
        
        if use_rag:
            # Retrieve context
            print("Retrieving context...")
            context = self.retrieve_context(query)
            
            # Format prompt with context
            prompt = self.prompt_template.format(
                context=context,
                question=query
            )
        else:
            # Direct prompt without RAG
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model"
        
        # Generate response
        print("Generating response...")
        
        with tf.device('/GPU:0'):
            response = self.model.generate(
                prompt,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                top_k=40
            )
        
        # Extract only the model's response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        
        return response.strip()
    
    def benchmark_performance(self):
        """Benchmark inference performance."""
        import time
        
        test_queries = [
            "What is our Q3 revenue?",
            "Explain our main strategic risks",
            "Describe Project Phoenix"
        ]
        
        print("\n" + "="*60)
        print("Performance Benchmark")
        print("="*60)
        
        for query in test_queries:
            start = time.time()
            response = self.generate_response(query)
            elapsed = time.time() - start
            
            tokens = len(response.split())
            tokens_per_sec = tokens / elapsed
            
            print(f"\nQuery: {query}")
            print(f"Response length: {tokens} tokens")
            print(f"Time: {elapsed:.2f}s")
            print(f"Speed: {tokens_per_sec:.1f} tokens/sec")
    
    def interactive_session(self):
        """Run interactive Q&A session."""
        print("\n" + "="*60)
        print("Gemma RAG-Powered Assistant (TensorFlow)")
        print("Commands: 'exit' to quit, 'benchmark' to test performance")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("\n💬 Your question: ").strip()
                
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if query.lower() == 'benchmark':
                    self.benchmark_performance()
                    continue
                
                if not query:
                    continue
                
                # Generate response
                print("\n🤖 Gemma: ", end='', flush=True)
                response = self.generate_response(query)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GemmaRAGPipeline(
        model_preset="gemma2_instruct_2b_en",  # Start with 2B
        checkpoint_path=None  # Add path to LoRA checkpoint if available
    )
    
    # Run interactive session
    pipeline.interactive_session()
```

## Utility Scripts

### System Monitor for TensorFlow
Create `scripts/monitor_tf_resources.py`:
```python
#!/usr/bin/env python3
"""Monitor TensorFlow and system resources."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import psutil
import time
from datetime import datetime

def monitor_tf_resources(interval=2):
    """Monitor TensorFlow Metal and system resources."""
    
    print("Monitoring TensorFlow Metal Resources (Ctrl+C to stop)")
    print("-" * 60)
    
    # Get GPU device
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    try:
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # GPU status
            gpu_status = "✅ Active" if gpus else "❌ Not found"
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Print stats
            print(f"[{timestamp}] "
                  f"CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {memory_used_gb:5.1f}/{memory_total_gb:5.1f}GB | "
                  f"Metal GPU: {gpu_status}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_tf_resources()
```

## Complete Test Suite
Create `scripts/test_pipeline_tf.py`:
```python
#!/usr/bin/env python3
"""Test the complete TensorFlow/Gemma pipeline."""

import os
import sys
from pathlib import Path

def run_tests():
    """Run all pipeline tests."""
    
    tests = [
        ("TensorFlow Metal", "python scripts/verify_tf_metal.py"),
        ("Document Processing", "python scripts/process_documents_tf.py"),
        ("Vector DB Setup", "python scripts/setup_vectordb_tf.py"),
        ("Fine-tuning Data", "python scripts/prepare_finetuning_tf.py"),
        ("Model Loading", "python scripts/setup_gemma.py"),
    ]
    
    print("="*60)
    print("TENSORFLOW/GEMMA PIPELINE TESTING")
    print("="*60)
    
    results = []
    for test_name, command in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-"*40)
        
        result = os.system(command)
        
        if result == 0:
            print(f"✅ {test_name}: PASSED")
            results.append((test_name, "PASSED"))
        else:
            print(f"❌ {test_name}: FAILED")
            results.append((test_name, "FAILED"))
    
    print("\n" + "="*60)
    print("Test Summary:")
    for name, status in results:
        print(f"  {name}: {status}")
    print("="*60)

if __name__ == "__main__":
    run_tests()
```

## Makefile
Create `Makefile`:
```makefile
# Makefile for TensorFlow/Gemma Local AI System

.PHONY: help setup test train inference benchmark clean

help:
	@echo "TensorFlow/Gemma Local AI System"
	@echo "================================"
	@echo "  make setup      - Set up TensorFlow environment"
	@echo "  make test       - Run all tests"
	@echo "  make train      - Run LoRA fine-tuning"
	@echo "  make inference  - Start RAG inference pipeline"
	@echo "  make benchmark  - Run performance benchmarks"
	@echo "  make monitor    - Monitor system resources"
	@echo "  make clean      - Clean temporary files"

setup:
	conda env create -f environment.yml
	conda activate tf_gemma
	python scripts/verify_tf_metal.py

test:
	python scripts/test_pipeline_tf.py

train:
	python scripts/prepare_finetuning_tf.py
	python scripts/finetune_gemma_lora.py

inference:
	python scripts/inference_pipeline_tf.py

benchmark:
	python -c "from scripts.inference_pipeline_tf import GemmaRAGPipeline; p = GemmaRAGPipeline(); p.benchmark_performance()"

monitor:
	python scripts/monitor_tf_resources.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf logs/*.log
```

## Quick Start Guide

### 1. Initial Setup (One-time)
```bash
cd /Users/whitneywalters/AIProgramming/tuning
source .env
make setup
```

### 2. Prepare Data
```bash
# Add your documents to business_docs/
# Then process them:
python scripts/process_documents_tf.py
python scripts/setup_vectordb_tf.py
```

### 3. Download Gemma Model
```bash
# Accept license at https://www.kaggle.com/models/google/gemma-2
python scripts/setup_gemma.py
```

### 4. Fine-tune (Optional)
```bash
make train
```

### 5. Run Inference
```bash
make inference
```

## Key Advantages of This Implementation

1. **Leverages Your TensorFlow Expertise** - No learning curve
2. **Stable Metal Support** - TensorFlow-Metal is mature and reliable
3. **Better Memory Management** - TensorFlow's graph optimization
4. **Production Ready** - Can deploy with TensorFlow Serving or TFLite
5. **Faster Development** - Use your existing debugging skills

## Performance Optimization Tips

1. **Use Mixed Precision**: Already configured in the scripts
2. **Batch Processing**: Adjust batch sizes based on available memory
3. **Model Caching**: TensorFlow automatically caches compiled graphs
4. **Gradient Checkpointing**: Enable for larger models

## Next Steps

1. Start with the 2B model for testing
2. Validate the pipeline end-to-end
3. Scale up to 9B model once verified
4. Add more training data for better fine-tuning
5. Optimize inference with TensorFlow Lite for deployment

This implementation is optimized for your TensorFlow expertise and M3 Pro hardware, providing a more stable and efficient path to production than the PyTorch alternative.