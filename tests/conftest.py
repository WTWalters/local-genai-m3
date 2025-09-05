"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# Add scripts to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "content": "Q3 2024 revenue was $45.2M with 15% YoY growth. Project Phoenix contributed $5M to overall revenue.",
            "metadata": {
                "source": "q3_report.md",
                "chunk_index": 0,
                "total_chunks": 2
            }
        },
        {
            "content": "Strategic risk assessment identifies cybersecurity as high priority with high impact and high probability.",
            "metadata": {
                "source": "risk_assessment.md", 
                "chunk_index": 0,
                "total_chunks": 1
            }
        },
        {
            "content": "Digital transformation initiatives have yielded 8% operational cost reduction through automation.",
            "metadata": {
                "source": "q3_report.md",
                "chunk_index": 1,
                "total_chunks": 2
            }
        }
    ]

@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    docs_dir = temp_dir / "business_docs"
    docs_dir.mkdir()
    
    # Sample markdown document
    sample_md = docs_dir / "sample.md"
    sample_md.write_text("""# Q3 2024 Report

## Revenue
Q3 2024 revenue was $45.2M with 15% YoY growth.

## Projects  
Project Phoenix contributed $5M to overall revenue.

## Efficiency
Digital transformation initiatives have yielded 8% operational cost reduction.
""")
    
    # Sample text document
    sample_txt = docs_dir / "risks.txt"
    sample_txt.write_text("""Strategic Risk Assessment

1. Cybersecurity: High impact, High probability
2. Supply chain: High impact, Medium probability
3. Regulatory changes: Medium impact, High probability
""")
    
    return docs_dir

@pytest.fixture
def sample_chunks_file(temp_dir, sample_chunks):
    """Create a sample chunks JSON file."""
    chunks_file = temp_dir / "processed_chunks.json"
    with open(chunks_file, 'w') as f:
        json.dump(sample_chunks, f)
    return chunks_file

@pytest.fixture
def mock_env_vars(temp_dir, monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
    monkeypatch.setenv("MODEL_PATH", str(temp_dir / "models"))
    monkeypatch.setenv("DATA_PATH", str(temp_dir / "data"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(temp_dir / "checkpoints"))
    
    # Create directories
    (temp_dir / "models").mkdir()
    (temp_dir / "data").mkdir() 
    (temp_dir / "checkpoints").mkdir()

@pytest.fixture
def skip_if_no_tf():
    """Skip test if TensorFlow is not available."""
    pytest.importorskip("tensorflow")

@pytest.fixture
def skip_if_no_torch():
    """Skip test if PyTorch is not available."""
    pytest.importorskip("torch")