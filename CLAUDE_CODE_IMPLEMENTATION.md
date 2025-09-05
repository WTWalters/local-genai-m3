# Claude Code Implementation Guide: Local Gen AI on MacBook M3

## Overview
This guide provides step-by-step instructions for implementing a local generative AI system on MacBook M3 using Claude Code. The implementation follows the blueprint architecture with necessary corrections and improvements.

## Prerequisites

### System Requirements
- MacBook Pro M3 (minimum 18GB RAM, ideally 24GB+)
- macOS 14.0 or later
- 50GB+ free disk space
- Xcode Command Line Tools installed
- Homebrew installed

### Initial Setup Commands
```bash
# Install Xcode Command Line Tools (if needed)
xcode-select --install

# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system dependencies
brew install cmake git git-lfs wget
```

## Phase 1: Environment Setup

### Step 1.1: Create Project Structure
```bash
cd /Users/whitneywalters/AIProgramming/tuning
mkdir -p {models,data,scripts,configs,checkpoints,logs,chroma_db,business_docs}

# Create a project README
echo "# Local GenAI System on M3" > README.md
```

### Step 1.2: Install Miniforge and Create Environment
```bash
# Install Miniforge for arm64
brew install miniforge

# Create conda environment with explicit channel
conda create -n genai_m3 python=3.11 -c conda-forge -y
conda activate genai_m3

# Create environment file for reproducibility
cat > environment.yml << 'EOF'
name: genai_m3
channels:
  - conda-forge
  - pytorch-nightly
dependencies:
  - python=3.11
  - pip
  - numpy
  - scipy
  - ipython
  - jupyter
  - pip:
    - --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    - transformers==4.44.0
    - accelerate==0.33.0
    - peft==0.12.0
    - datasets==2.20.0
    - sentence-transformers==3.0.1
    - chromadb==0.5.5
    - docling==2.48.0
    - langchain==0.2.14
    - langchain-community==0.2.12
    - huggingface-hub==0.24.6
    - llama-cpp-python==0.2.90
    - tqdm
    - pandas
    - openpyxl
EOF

# Install from environment file
conda env update -f environment.yml
```

### Step 1.3: Verify MPS Support
Create `scripts/verify_mps.py`:
```python
#!/usr/bin/env python3
"""Verify MPS support and PyTorch installation."""

import sys
import torch
import platform

def verify_environment():
    """Comprehensive environment verification."""
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # System info
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print()
    
    # PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        # Test MPS with a simple operation
        try:
            device = torch.device("mps")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"MPS Test: SUCCESS (computed 1000x1000 matrix multiplication)")
            print(f"Device: {z.device}")
        except Exception as e:
            print(f"MPS Test: FAILED - {e}")
    else:
        print("WARNING: MPS not available. Check PyTorch installation.")
    
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
```

Run verification:
```bash
chmod +x scripts/verify_mps.py
python scripts/verify_mps.py
```

### Step 1.4: Set Environment Variables
Create `.env` file:
```bash
cat > .env << 'EOF'
# MPS Configuration
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Hugging Face Configuration
export HF_HOME=/Users/whitneywalters/AIProgramming/tuning/.cache/huggingface
export TRANSFORMERS_CACHE=/Users/whitneywalters/AIProgramming/tuning/.cache/transformers

# Project Paths
export PROJECT_ROOT=/Users/whitneywalters/AIProgramming/tuning
export MODEL_PATH=$PROJECT_ROOT/models
export DATA_PATH=$PROJECT_ROOT/data
export CHECKPOINT_PATH=$PROJECT_ROOT/checkpoints

# Memory Management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
EOF

# Source the environment
source .env
```

## Phase 2: Model Setup

### Step 2.1: Build llama.cpp with Metal Support
```bash
cd $PROJECT_ROOT

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with Metal support
make clean
make LLAMA_METAL=1 -j8

# Verify the build
./main --help | grep -i metal
```

### Step 2.2: Download Base Model
Create `scripts/download_model.py`:
```python
#!/usr/bin/env python3
"""Download and prepare Llama 3 8B model."""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model():
    """Download Llama 3 8B Instruct model."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_dir = Path(os.environ.get("MODEL_PATH", "./models")) / "llama3-8b-instruct"
    
    print(f"Downloading {model_id} to {local_dir}")
    
    # Note: You'll need to authenticate with Hugging Face
    # Run: huggingface-cli login
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Model downloaded successfully to {local_dir}")
        return str(local_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Make sure you're logged in to Hugging Face and have access to the model.")
        return None

if __name__ == "__main__":
    download_model()
```

### Step 2.3: Convert to GGUF Format
Create `scripts/convert_to_gguf.py`:
```python
#!/usr/bin/env python3
"""Convert Hugging Face model to GGUF format."""

import os
import subprocess
from pathlib import Path

def convert_to_gguf(model_path, quantization="Q4_K_M"):
    """Convert HF model to GGUF format."""
    
    llama_cpp_path = Path(os.environ.get("PROJECT_ROOT", ".")) / "llama.cpp"
    output_path = Path(os.environ.get("MODEL_PATH", "./models")) / f"llama3-8b-instruct.{quantization}.gguf"
    
    # Install required Python packages for conversion
    subprocess.run([
        "pip", "install", "-r", 
        str(llama_cpp_path / "requirements.txt")
    ], check=True)
    
    # Convert to GGUF
    convert_script = llama_cpp_path / "convert.py"
    
    cmd = [
        "python", str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization.lower()
    ]
    
    print(f"Converting model to GGUF format ({quantization})...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Model converted successfully: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return None

if __name__ == "__main__":
    model_path = Path(os.environ.get("MODEL_PATH", "./models")) / "llama3-8b-instruct"
    
    # Convert to multiple quantization levels for testing
    for quant in ["Q4_K_M", "Q5_K_M", "Q6_K"]:
        convert_to_gguf(model_path, quant)
```

## Phase 3: RAG Pipeline Setup

### Step 3.1: Document Processing with Docling
Create `scripts/process_documents.py`:
```python
#!/usr/bin/env python3
"""Process documents using Docling for RAG pipeline."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self, corpus_path: str = "./business_docs"):
        self.corpus_path = Path(corpus_path)
        self.converter = DocumentConverter()
        
        # Initialize chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # For semantic chunking logic
        self.chunker_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document."""
        try:
            result = self.converter.convert(str(file_path))
            
            # Export to markdown format
            markdown_content = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                "source": file_path.name,
                "path": str(file_path),
                "format": file_path.suffix,
                "page_count": getattr(result.document, 'page_count', None),
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
    
    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a processed document."""
        if doc_data["status"] != "success" or not doc_data["content"]:
            return []
        
        chunks = self.text_splitter.split_text(doc_data["content"])
        
        return [
            {
                "content": chunk,
                "metadata": {
                    **doc_data["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def process_corpus(self) -> List[Dict[str, Any]]:
        """Process all documents in the corpus."""
        all_chunks = []
        
        # Get all documents
        doc_files = list(self.corpus_path.glob("**/*"))
        doc_files = [f for f in doc_files if f.suffix in [".pdf", ".docx", ".txt", ".md"]]
        
        print(f"Found {len(doc_files)} documents to process")
        
        for file_path in tqdm(doc_files, desc="Processing documents"):
            doc_data = self.process_document(file_path)
            chunks = self.chunk_document(doc_data)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(doc_files)} documents")
        
        # Save chunks for inspection
        output_path = Path("data/processed_chunks.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(all_chunks, f, indent=2)
        
        return all_chunks

if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_corpus()
```

### Step 3.2: Vector Database Setup
Create `scripts/setup_vectordb.py`:
```python
#!/usr/bin/env python3
"""Setup ChromaDB vector database with embeddings."""

import os
import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class VectorDBManager:
    """Manage ChromaDB vector database."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "business_knowledge"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Initialize embedding model (nomic-embed-text-v1.5)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Business knowledge base for RAG"}
            )
            print(f"Created new collection: {self.collection_name}")
        except:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Add document chunks to the vector database."""
        
        # Prepare data
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i:06d}" for i in range(len(chunks))]
        
        # Process in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector DB"):
            batch_texts = texts[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Generate embeddings with task prefix for RAG
            batch_texts_with_prefix = [f"search_document: {text}" for text in batch_texts]
            batch_embeddings = self.embedding_model.encode(
                batch_texts_with_prefix,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to collection
            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        
        print(f"Added {len(chunks)} chunks to the vector database")
        print(f"Total documents in collection: {self.collection.count()}")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search the vector database."""
        
        # Add query prefix for nomic model
        query_with_prefix = f"search_query: {query}"
        query_embedding = self.embedding_model.encode(
            query_with_prefix,
            convert_to_numpy=True
        )
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "db_path": str(self.db_path)
        }

if __name__ == "__main__":
    # Load processed chunks
    chunks_path = Path("data/processed_chunks.json")
    
    if not chunks_path.exists():
        print("No processed chunks found. Run process_documents.py first.")
        exit(1)
    
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    # Setup vector database
    db_manager = VectorDBManager()
    db_manager.add_documents(chunks)
    
    # Test search
    test_query = "What are the key business objectives?"
    results = db_manager.search(test_query, n_results=3)
    
    print("\nTest Search Results:")
    print(f"Query: {test_query}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"   Content: {result['content'][:200]}...")
```

## Phase 4: Fine-Tuning Pipeline

### Step 4.1: Dataset Preparation
Create `scripts/prepare_finetuning_data.py`:
```python
#!/usr/bin/env python3
"""Prepare fine-tuning dataset for business-specific adaptation."""

import json
from pathlib import Path
from typing import List, Dict, Any
import random

class FinetuningDatasetBuilder:
    """Build instruction-tuning dataset."""
    
    def __init__(self, output_path: str = "data/finetuning_dataset.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True)
        
        # Llama 3 instruction format
        self.instruction_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful expert assistant for our company. Use the provided context to answer questions accurately and professionally.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
    
    def create_golden_examples(self) -> List[Dict[str, str]]:
        """Create high-quality manual examples."""
        
        # These should be replaced with actual business-specific examples
        examples = [
            {
                "instruction": "Summarize the Q3 financial report for the executive team",
                "context": "Q3 2024 Financial Results: Revenue: $45.2M (15% YoY growth), Operating Income: $8.3M, Key drivers: Product line expansion (Phoenix project contributed $5M), International market growth (+22%), Digital transformation initiatives reducing costs by 8%.",
                "response": "Q3 2024 demonstrated strong performance with revenue reaching $45.2M, representing 15% year-over-year growth. The successful launch of Project Phoenix contributed $5M to revenues, while our international expansion drove 22% growth in overseas markets. Additionally, our digital transformation initiatives have yielded an 8% reduction in operational costs, resulting in operating income of $8.3M."
            },
            {
                "instruction": "What are the main risks identified in our strategic planning?",
                "context": "Strategic Risk Assessment 2024: 1) Supply chain disruptions (High impact, Medium probability), 2) Cybersecurity threats (High impact, High probability), 3) Regulatory compliance changes (Medium impact, High probability), 4) Talent retention (Medium impact, Medium probability), 5) Market competition from new entrants (Low impact, High probability).",
                "response": "Our strategic risk assessment identifies five key areas of concern: The highest priority is cybersecurity threats, rated as both high impact and high probability. Supply chain disruptions pose significant risk with high potential impact. Regulatory compliance changes require close monitoring given their high probability. Talent retention and new market entrants represent moderate risks that need ongoing attention and mitigation strategies."
            }
        ]
        
        return examples
    
    def format_for_training(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format examples for training."""
        formatted = []
        
        for ex in examples:
            text = self.instruction_template.format(
                context=ex["context"],
                question=ex["instruction"],
                response=ex["response"]
            )
            
            formatted.append({
                "text": text,
                "instruction": ex["instruction"],
                "context": ex["context"],
                "response": ex["response"]
            })
        
        return formatted
    
    def save_dataset(self, examples: List[Dict[str, str]]):
        """Save dataset to JSON."""
        with open(self.output_path, "w") as f:
            json.dump(examples, f, indent=2)
        
        print(f"Saved {len(examples)} examples to {self.output_path}")
    
    def build_dataset(self):
        """Build complete dataset."""
        # Start with golden examples
        golden = self.create_golden_examples()
        
        # Format for training
        formatted = self.format_for_training(golden)
        
        # Save
        self.save_dataset(formatted)
        
        return formatted

if __name__ == "__main__":
    builder = FinetuningDatasetBuilder()
    dataset = builder.build_dataset()
```

### Step 4.2: LoRA Fine-Tuning
Create `scripts/finetune_lora.py`:
```python
#!/usr/bin/env python3
"""Fine-tune Llama 3 with LoRA on MPS."""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

class LoRAFineTuner:
    """Fine-tune model with LoRA."""
    
    def __init__(self, model_path: str, output_dir: str = "./checkpoints/lora"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        print("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model in float16 for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded: {self.model.config.model_type}")
    
    def prepare_model_for_lora(self):
        """Apply LoRA configuration."""
        
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target layers
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, dataset_path: str):
        """Prepare training dataset."""
        
        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, dataset_path: str):
        """Run training."""
        
        # Load components
        self.load_model_and_tokenizer()
        self.prepare_model_for_lora()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset_path)
        
        # Training arguments optimized for M3
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch for M3
            gradient_accumulation_steps=4,  # Accumulate gradients
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,  # Mixed precision
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
            use_mps_device=True,  # Use MPS
            dataloader_num_workers=0,  # Avoid multiprocessing issues on macOS
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("Training complete!")

if __name__ == "__main__":
    model_path = Path(os.environ.get("MODEL_PATH", "./models")) / "llama3-8b-instruct"
    dataset_path = "data/finetuning_dataset.json"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}. Run prepare_finetuning_data.py first.")
        exit(1)
    
    tuner = LoRAFineTuner(model_path=str(model_path))
    tuner.train(dataset_path)
```

## Phase 5: Inference Pipeline

### Step 5.1: RAG + Fine-tuned Model Integration
Create `scripts/inference_pipeline.py`:
```python
#!/usr/bin/env python3
"""Complete inference pipeline with RAG and fine-tuned model."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama
import chromadb
from sentence_transformers import SentenceTransformer

class RAGInferencePipeline:
    """Complete RAG inference pipeline."""
    
    def __init__(
        self,
        model_path: str,
        db_path: str = "./chroma_db",
        collection_name: str = "business_knowledge"
    ):
        # Load GGUF model
        print(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all GPU layers
            n_ctx=4096,  # Context window
            n_batch=512,
            verbose=False
        )
        
        # Load embedding model for queries
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        
        # Connect to ChromaDB
        print("Connecting to vector database...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        # Prompt template
        self.prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful expert assistant for our company. Use the following context to answer the question. 
Answer with a professional and analytical tone. If the answer is not found in the context, 
state that you don't have enough information.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from vector DB."""
        
        # Add query prefix for nomic model
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
    
    def generate_response(self, query: str, stream: bool = True) -> str:
        """Generate response using RAG."""
        
        # Retrieve context
        print("Retrieving context...")
        context = self.retrieve_context(query)
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # Generate response
        print("Generating response...")
        
        if stream:
            # Streaming response
            response = self.llm(
                prompt,
                max_tokens=1024,
                stop=["<|eot_id|>"],
                echo=False,
                stream=True,
                temperature=0.7
            )
            
            full_response = ""
            for chunk in response:
                token = chunk['choices'][0]['text']
                full_response += token
                print(token, end='', flush=True)
            
            print()  # New line after response
            return full_response
        else:
            # Non-streaming response
            response = self.llm(
                prompt,
                max_tokens=1024,
                stop=["<|eot_id|>"],
                echo=False,
                temperature=0.7
            )
            
            return response['choices'][0]['text']
    
    def interactive_session(self):
        """Run interactive Q&A session."""
        print("\n" + "="*60)
        print("RAG-Powered AI Assistant")
        print("Type 'exit' to quit, 'context' to see retrieved context")
        print("="*60 + "\n")
        
        self.show_context = False
        
        while True:
            try:
                query = input("\n📝 Your question: ").strip()
                
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if query.lower() == 'context':
                    self.show_context = not self.show_context
                    print(f"Context display: {'ON' if self.show_context else 'OFF'}")
                    continue
                
                if not query:
                    continue
                
                # Show retrieved context if enabled
                if self.show_context:
                    context = self.retrieve_context(query)
                    print("\n📚 Retrieved Context:")
                    print("-" * 40)
                    print(context)
                    print("-" * 40)
                
                # Generate response
                print("\n🤖 Assistant: ", end='')
                response = self.generate_response(query, stream=True)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Path to GGUF model
    model_path = Path(os.environ.get("MODEL_PATH", "./models")) / "llama3-8b-instruct.Q4_K_M.gguf"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run the model conversion script first.")
        exit(1)
    
    # Initialize pipeline
    pipeline = RAGInferencePipeline(
        model_path=str(model_path)
    )
    
    # Run interactive session
    pipeline.interactive_session()
```

## Utility Scripts

### Monitor System Resources
Create `scripts/monitor_resources.py`:
```python
#!/usr/bin/env python3
"""Monitor system resources during training/inference."""

import psutil
import subprocess
import time
from datetime import datetime

def get_gpu_memory():
    """Get GPU memory usage on macOS."""
    try:
        # Use system_profiler to get Metal GPU info
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        return "GPU info not available"

def monitor_resources(interval=2):
    """Monitor system resources."""
    print("Monitoring system resources (Ctrl+C to stop)...")
    print("-" * 60)
    
    try:
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Print stats
            print(f"[{timestamp}] CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {memory_used_gb:5.1f}/{memory_total_gb:5.1f}GB ({memory_percent:5.1f}%)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_resources()
```

## Testing Scripts

### Test Complete Pipeline
Create `scripts/test_pipeline.py`:
```python
#!/usr/bin/env python3
"""Test the complete pipeline."""

import os
import sys
from pathlib import Path

def run_tests():
    """Run pipeline tests."""
    
    tests = [
        ("MPS Support", "python scripts/verify_mps.py"),
        ("Document Processing", "python scripts/process_documents.py"),
        ("Vector DB Setup", "python scripts/setup_vectordb.py"),
        ("Fine-tuning Data", "python scripts/prepare_finetuning_data.py"),
    ]
    
    print("="*60)
    print("PIPELINE TESTING")
    print("="*60)
    
    for test_name, command in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-"*40)
        
        result = os.system(command)
        
        if result == 0:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
            
    print("\n" + "="*60)
    print("Testing complete!")

if __name__ == "__main__":
    run_tests()
```

## Makefile for Easy Management
Create `Makefile`:
```makefile
# Makefile for Local GenAI System

.PHONY: help setup test clean run monitor

help:
	@echo "Available commands:"
	@echo "  make setup    - Set up environment and download models"
	@echo "  make test     - Run all tests"
	@echo "  make run      - Run inference pipeline"
	@echo "  make monitor  - Monitor system resources"
	@echo "  make clean    - Clean temporary files"

setup:
	conda env create -f environment.yml
	python scripts/verify_mps.py
	@echo "Setup complete! Activate environment with: conda activate genai_m3"

test:
	python scripts/test_pipeline.py

run:
	python scripts/inference_pipeline.py

monitor:
	python scripts/monitor_resources.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
```

## Next Steps for Claude Code

1. **Start with Environment Setup**
   ```bash
   cd /Users/whitneywalters/AIProgramming/tuning
   make setup
   ```

2. **Download and Convert Model**
   - Run `python scripts/download_model.py`
   - Run `python scripts/convert_to_gguf.py`

3. **Process Documents**
   - Add documents to `business_docs/` folder
   - Run `python scripts/process_documents.py`
   - Run `python scripts/setup_vectordb.py`

4. **Fine-tune (Optional)**
   - Create training data with `python scripts/prepare_finetuning_data.py`
   - Run fine-tuning with `python scripts/finetune_lora.py`

5. **Run Inference**
   ```bash
   make run
   ```

## Important Notes

- Always activate the conda environment before running scripts
- Monitor GPU memory usage during training/inference
- Start with smaller batch sizes and adjust based on available memory
- Test each component independently before running the full pipeline
- Keep backups of successful model checkpoints

This implementation guide provides a complete, production-ready system with proper error handling, monitoring, and testing capabilities.