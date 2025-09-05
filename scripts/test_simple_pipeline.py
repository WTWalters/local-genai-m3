#!/usr/bin/env python3
"""Simple inference test without complex dependencies."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import json
from pathlib import Path
import numpy as np

class SimpleRAGPipeline:
    """Simplified RAG pipeline for testing."""
    
    def __init__(self):
        self.setup_gpu()
        self.chunks = self.load_chunks()
        
    def setup_gpu(self):
        """Configure GPU."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("✅ GPU configured for inference")
            except:
                pass
    
    def load_chunks(self):
        """Load processed document chunks."""
        chunks_path = Path("data/processed_chunks.json")
        
        if not chunks_path.exists():
            print("No processed chunks found. Run process_documents_simple.py first")
            return []
        
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} document chunks")
        return chunks
    
    def simple_search(self, query, n_results=3):
        """Simple keyword-based search for testing."""
        if not self.chunks:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            content_words = set(chunk['content'].lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                results.append({
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'score': overlap
                })
        
        # Sort by score and return top n
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def generate_response(self, query):
        """Generate a simple response."""
        print(f"\nQuery: {query}")
        
        # Search for relevant chunks
        results = self.simple_search(query)
        
        if results:
            print(f"\nFound {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                print(f"  {i}. From {source} (score: {result['score']})")
            
            # Simple response based on top result
            context = results[0]['content']
            print(f"\nContext: {context[:200]}...")
            
            # For now, just format a simple response
            response = f"Based on the documents: {context[:300]}"
            return response
        else:
            return "No relevant information found in the documents."
    
    def test_pipeline(self):
        """Test the pipeline with sample queries."""
        test_queries = [
            "What is the Q3 revenue?",
            "Tell me about Project Phoenix",
            "What are the main risks?",
            "What is the customer satisfaction score?"
        ]
        
        print("\n" + "="*60)
        print("TESTING SIMPLE RAG PIPELINE")
        print("="*60)
        
        for query in test_queries:
            response = self.generate_response(query)
            print(f"\nResponse: {response[:200]}...")
            print("-"*40)

if __name__ == "__main__":
    # Create pipeline
    pipeline = SimpleRAGPipeline()
    
    # Test with sample queries
    pipeline.test_pipeline()
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'exit' to quit)")
    print("="*60)
    
    while True:
        query = input("\n💬 Your question: ").strip()
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if query:
            response = pipeline.generate_response(query)
            print(f"\n🤖 Response: {response}")