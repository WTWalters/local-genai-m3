# Local GenAI System - Dual Implementation Approach

A comprehensive local generative AI system optimized for MacBook M3 Pro, featuring both PyTorch/Llama and TensorFlow/Gemma implementations with RAG (Retrieval-Augmented Generation) capabilities.

## 🏗️ Architecture Overview

This project implements a complete local AI stack with:
- **Document Processing**: Docling-based PDF/document parsing
- **Vector Database**: ChromaDB with nomic embeddings
- **RAG Pipeline**: Semantic search and context retrieval
- **Fine-Tuning**: LoRA adaptation for domain-specific knowledge
- **Inference**: Local model serving with Metal acceleration

## 🚀 Two Implementation Paths

### Option 1: PyTorch + Llama 3 (Cutting Edge)
- **Framework**: PyTorch with MPS backend
- **Model**: Llama 3 8B Instruct
- **Quantization**: GGUF via llama.cpp
- **Fine-tuning**: PEFT/LoRA
- **Best for**: Latest research, extensive ecosystem

### Option 2: TensorFlow + Gemma 2 (Recommended)
- **Framework**: TensorFlow with Metal plugin
- **Model**: Gemma 2 9B/2B Instruct
- **Quantization**: GGUF via llama.cpp
- **Fine-tuning**: Keras NLP LoRA
- **Best for**: Production stability, TensorFlow expertise

## 📋 Requirements

### Hardware
- MacBook M3 Pro (36GB RAM recommended)
- 50GB+ free disk space
- macOS 14.0+

### Software
- Python 3.11+
- Conda/Miniforge
- Xcode Command Line Tools

## 🛠️ Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd tuning

# Choose your implementation path:

# For TensorFlow/Gemma (Recommended):
conda env create -f environment.yml -n tf_gemma
conda activate tf_gemma

# For PyTorch/Llama:
conda env create -f environment.yml -n genai_m3
conda activate genai_m3

# Verify setup
python scripts/verify_tf_metal.py  # or verify_mps.py for PyTorch
```

### 2. Process Documents
```bash
# Add your business documents to business_docs/
mkdir -p business_docs
# <copy your PDFs, docs, etc.>

# Process documents
python scripts/process_documents_tf.py  # or process_documents.py
python scripts/setup_vectordb_tf.py     # or setup_vectordb.py
```

### 3. Download Models
```bash
# For Gemma (requires Kaggle account):
python scripts/setup_gemma.py

# For Llama (requires HuggingFace access):
python scripts/download_model.py
python scripts/convert_to_gguf.py
```

### 4. Fine-tune (Optional)
```bash
# Prepare training data
python scripts/prepare_finetuning_tf.py  # or prepare_finetuning_data.py

# Run fine-tuning
python scripts/finetune_gemma_lora.py    # or finetune_lora.py
```

### 5. Run Inference
```bash
# Start RAG-enhanced chat
python scripts/inference_pipeline_tf.py  # or inference_pipeline.py
```

## 📁 Project Structure

```
tuning/
├── scripts/                    # Implementation scripts
│   ├── verify_tf_metal.py     # TensorFlow Metal verification
│   ├── verify_mps.py          # PyTorch MPS verification
│   ├── process_documents*.py  # Document processing
│   ├── setup_vectordb*.py     # Vector database setup
│   ├── finetune_*.py          # Fine-tuning scripts
│   └── inference_pipeline*.py # Inference pipelines
├── models/                     # Downloaded/converted models
├── data/                      # Processed datasets
├── business_docs/             # Your source documents
├── checkpoints/              # Fine-tuning checkpoints
├── chroma_db/               # Vector database
├── tests/                   # Unit tests
├── docs/                   # Technical documentation
└── Makefile                # Automation commands
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_document_processing.py -v
python -m pytest tests/test_rag_pipeline.py -v
python -m pytest tests/test_model_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html
```

## 📊 Performance Benchmarks

### Expected Performance (M3 Pro 36GB)

| Task | TensorFlow/Gemma | PyTorch/Llama |
|------|------------------|---------------|
| Inference Speed | 25-40 tokens/sec | 20-35 tokens/sec |
| Memory Usage | ~30GB peak | ~32GB peak |
| Training Stability | High | Medium |
| Setup Time | 2 hours | 4-6 hours |

## 🔧 Makefile Commands

```bash
make help       # Show all available commands
make setup      # Environment setup
make test       # Run all tests
make train      # Run fine-tuning pipeline
make inference  # Start inference server
make benchmark  # Performance benchmarks
make monitor    # System resource monitoring
make clean      # Clean temporary files
```

## 📖 Documentation

- [Technical Review - PyTorch Approach](DOCUMENT_REVIEW.md)
- [Technical Review - TensorFlow Approach](DOCUMENT_REVIEW_TENSORFLOW.md)
- [Implementation Guide - PyTorch](CLAUDE_CODE_IMPLEMENTATION.md)
- [Implementation Guide - TensorFlow](CLAUDE_CODE_IMPLEMENTATION_TENSORFLOW.md)

## 🐛 Troubleshooting

### Common Issues

1. **Metal GPU not detected**
   ```bash
   # For TensorFlow
   pip install tensorflow-metal==1.1.0
   
   # For PyTorch
   pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. **Memory issues during training**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use smaller model variant (2B instead of 9B)

3. **Model download failures**
   - Verify Kaggle/HuggingFace authentication
   - Check internet connection
   - Try downloading manually

### Getting Help

1. Check the troubleshooting section in implementation guides
2. Run diagnostic scripts: `python scripts/verify_*.py`
3. Monitor resources: `make monitor`
4. Check logs in `logs/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IBM Docling for document processing
- Nomic AI for embedding models
- Google for Gemma models
- Meta for Llama models
- ChromaDB for vector storage
- llama.cpp for efficient inference

---

**Note**: This implementation is optimized for Apple Silicon (M1/M2/M3) and leverages Metal Performance Shaders for GPU acceleration. For other platforms, modifications may be required.