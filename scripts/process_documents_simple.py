#!/usr/bin/env python3
"""Simple document processing for RAG pipeline without TensorFlow dependencies."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

class SimpleDocumentProcessor:
    """Process documents with basic text splitting."""
    
    def __init__(self, corpus_path: str = "./business_docs"):
        self.corpus_path = Path(corpus_path)
        
    def read_markdown_file(self, file_path: Path) -> str:
        """Read markdown file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def chunk_text_simple(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Simple text chunking by sentences and paragraphs."""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Clean paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 100]
        
        return chunks
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document."""
        
        # Only handle markdown files for simplicity
        if file_path.suffix != '.md':
            return {
                "content": None,
                "metadata": {"source": file_path.name},
                "status": "skipped",
                "reason": "Not a markdown file"
            }
        
        content = self.read_markdown_file(file_path)
        if not content:
            return {
                "content": None,
                "metadata": {"source": file_path.name},
                "status": "error",
                "reason": "Could not read file"
            }
        
        return {
            "content": content,
            "metadata": {
                "source": file_path.name,
                "path": str(file_path),
                "format": file_path.suffix,
                "size": len(content)
            },
            "status": "success"
        }
    
    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a processed document."""
        if doc_data["status"] != "success" or not doc_data["content"]:
            return []
        
        chunks = self.chunk_text_simple(doc_data["content"])
        
        return [
            {
                "content": chunk,
                "metadata": {
                    **doc_data["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def process_corpus(self) -> List[Dict[str, Any]]:
        """Process all documents in the corpus."""
        all_chunks = []
        
        # Get all markdown files
        doc_files = list(self.corpus_path.glob("*.md"))
        
        print(f"Found {len(doc_files)} markdown documents to process")
        
        for file_path in tqdm(doc_files, desc="Processing documents"):
            print(f"\nProcessing: {file_path.name}")
            
            doc_data = self.process_document(file_path)
            
            if doc_data["status"] == "success":
                chunks = self.chunk_document(doc_data)
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks")
            else:
                print(f"  {doc_data['status']}: {doc_data.get('reason', 'Unknown error')}")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Save chunks for inspection
        output_path = Path("data/processed_chunks.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"Chunks saved to: {output_path}")
        
        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"\nChunk {i+1} ({chunk['metadata']['source']}):")
            print(f"Size: {len(chunk['content'])} characters")
            print(f"Preview: {chunk['content'][:200]}...")
        
        return all_chunks
    
    def analyze_corpus(self):
        """Analyze the corpus for insights."""
        doc_files = list(self.corpus_path.glob("*.md"))
        
        print("=" * 60)
        print("CORPUS ANALYSIS")
        print("=" * 60)
        
        total_size = 0
        word_counts = []
        
        for file_path in doc_files:
            content = self.read_markdown_file(file_path)
            if content:
                size = len(content)
                words = len(content.split())
                total_size += size
                word_counts.append(words)
                
                print(f"\n{file_path.name}:")
                print(f"  Size: {size:,} characters")
                print(f"  Words: {words:,}")
                first_line = content.split('\n')[0][:80]
                print(f"  First line: {first_line}...")
        
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            print(f"\nCorpus Summary:")
            print(f"  Documents: {len(doc_files)}")
            print(f"  Total size: {total_size:,} characters")
            print(f"  Total words: {sum(word_counts):,}")
            print(f"  Average words per document: {avg_words:.0f}")
        
        print("=" * 60)

if __name__ == "__main__":
    processor = SimpleDocumentProcessor()
    
    # First analyze the corpus
    processor.analyze_corpus()
    
    # Then process it
    chunks = processor.process_corpus()