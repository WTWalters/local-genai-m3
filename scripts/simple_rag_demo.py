#!/usr/bin/env python3
"""Simple RAG demo using basic text similarity without complex dependencies."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import math

class SimpleRAG:
    """Simple RAG implementation using TF-IDF similarity."""
    
    def __init__(self, chunks_path: str = "data/processed_chunks.json"):
        self.chunks_path = Path(chunks_path)
        self.chunks = []
        self.vocabulary = set()
        self.idf_scores = {}
        
        if self.chunks_path.exists():
            self.load_chunks()
            self.build_index()
        else:
            print(f"❌ Chunks file not found: {chunks_path}")
            print("   Run process_documents_simple.py first")
    
    def load_chunks(self):
        """Load processed document chunks."""
        print(f"Loading chunks from {self.chunks_path}...")
        
        with open(self.chunks_path, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        
        # Show sample
        if self.chunks:
            sample = self.chunks[0]
            print(f"Sample chunk from '{sample['metadata']['source']}':")
            print(f"  Size: {len(sample['content'])} characters")
            print(f"  Preview: {sample['content'][:100]}...")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and remove empty strings
        words = [word for word in text.split() if word and len(word) > 2]
        
        return words
    
    def compute_tf(self, words: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        word_count = len(words)
        if word_count == 0:
            return {}
        
        tf_scores = {}
        word_freq = Counter(words)
        
        for word, count in word_freq.items():
            tf_scores[word] = count / word_count
        
        return tf_scores
    
    def build_index(self):
        """Build TF-IDF index for all chunks."""
        print("Building search index...")
        
        # Extract all words to build vocabulary
        all_words = set()
        chunk_words = []
        
        for chunk in self.chunks:
            words = self.preprocess_text(chunk['content'])
            chunk_words.append(words)
            all_words.update(words)
        
        self.vocabulary = all_words
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        
        # Compute IDF scores
        doc_count = len(self.chunks)
        for word in self.vocabulary:
            # Count documents containing this word
            docs_with_word = sum(1 for words in chunk_words if word in words)
            
            # Compute IDF (with smoothing)
            self.idf_scores[word] = math.log(doc_count / (1 + docs_with_word))
        
        # Store preprocessed words with each chunk
        for i, chunk in enumerate(self.chunks):
            chunk['_words'] = chunk_words[i]
            chunk['_tf'] = self.compute_tf(chunk_words[i])
        
        print("✅ Search index built")
    
    def compute_similarity(self, query_words: List[str], chunk: Dict) -> float:
        """Compute TF-IDF cosine similarity between query and chunk."""
        
        # Compute query TF scores
        query_tf = self.compute_tf(query_words)
        
        # Compute TF-IDF vectors
        query_vector = []
        chunk_vector = []
        
        # Use union of query and chunk words
        all_words = set(query_words) | set(chunk.get('_words', []))
        
        for word in all_words:
            # Query TF-IDF
            query_tf_score = query_tf.get(word, 0)
            query_idf_score = self.idf_scores.get(word, 0)
            query_tfidf = query_tf_score * query_idf_score
            query_vector.append(query_tfidf)
            
            # Chunk TF-IDF
            chunk_tf_score = chunk.get('_tf', {}).get(word, 0)
            chunk_tfidf = chunk_tf_score * query_idf_score  # Use same IDF
            chunk_vector.append(chunk_tfidf)
        
        # Compute cosine similarity
        if not query_vector or not chunk_vector:
            return 0.0
        
        dot_product = sum(q * c for q, c in zip(query_vector, chunk_vector))
        
        query_norm = math.sqrt(sum(q * q for q in query_vector))
        chunk_norm = math.sqrt(sum(c * c for c in chunk_vector))
        
        if query_norm == 0 or chunk_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * chunk_norm)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        
        if not self.chunks:
            print("❌ No chunks loaded")
            return []
        
        print(f"🔍 Searching for: '{query}'")
        
        # Preprocess query
        query_words = self.preprocess_text(query)
        
        if not query_words:
            print("❌ No valid words in query")
            return []
        
        # Compute similarities
        results = []
        for i, chunk in enumerate(self.chunks):
            similarity = self.compute_similarity(query_words, chunk)
            
            if similarity > 0:  # Only include chunks with some similarity
                results.append({
                    'chunk_id': i,
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'similarity': similarity
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        top_results = results[:n_results]
        
        print(f"✅ Found {len(top_results)} relevant results")
        
        return top_results
    
    def format_context(self, search_results: List[Dict]) -> str:
        """Format search results into context for generation."""
        
        if not search_results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result['metadata'].get('source', 'Unknown')
            content = result['content'].strip()
            similarity = result.get('similarity', 0)
            
            context_parts.append(f"[Source {i}: {source}] (Relevance: {similarity:.3f})\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieved context."""
        
        # Search for relevant chunks
        search_results = self.search(question, n_results=3)
        
        # Format context
        context = self.format_context(search_results)
        
        # Create a simple prompt (without actual LLM generation)
        prompt = f"""Based on the following context, please answer this question: {question}

Context:
{context}

Answer: [This would be generated by Gemma model]"""
        
        return {
            'question': question,
            'context': context,
            'search_results': search_results,
            'prompt': prompt
        }
    
    def interactive_demo(self):
        """Run interactive RAG demo."""
        
        # Check if running interactively
        import sys
        if not sys.stdin.isatty():
            print("❌ This demo requires interactive input. Skipping interactive mode.")
            return
        
        print("\n" + "="*60)
        print("SIMPLE RAG DEMO")
        print("Ask questions about the business documents!")
        print("Type 'exit' to quit, 'stats' for statistics")
        print("="*60)
        
        if not self.chunks:
            print("❌ No documents loaded. Cannot run demo.")
            return
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    print(f"\n📊 Statistics:")
                    print(f"  Total chunks: {len(self.chunks)}")
                    print(f"  Vocabulary size: {len(self.vocabulary)}")
                    print(f"  Average chunk size: {sum(len(c['content']) for c in self.chunks) / len(self.chunks):.0f} chars")
                    continue
                
                if not question:
                    continue
                
                # Get answer
                result = self.answer_question(question)
                
                print(f"\n📄 Context found:")
                if result['search_results']:
                    for i, res in enumerate(result['search_results'], 1):
                        source = res['metadata']['source']
                        similarity = res['similarity']
                        preview = res['content'][:100].replace('\n', ' ')
                        print(f"  {i}. {source} (sim: {similarity:.3f})")
                        print(f"     {preview}...")
                else:
                    print("  No relevant context found")
                
                print(f"\n🤖 Generated Prompt:")
                print("-" * 40)
                print(result['prompt'][:500] + "..." if len(result['prompt']) > 500 else result['prompt'])
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function."""
    
    print("="*60)
    print("SIMPLE RAG SYSTEM DEMO")  
    print("="*60)
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    if rag.chunks:
        # Run some test searches
        test_questions = [
            "What is our Q3 revenue?",
            "Tell me about Project Phoenix",
            "What are the main strategic risks?",
            "How much did digital transformation save us?",
            "What is our customer growth rate?"
        ]
        
        print(f"\n🧪 Testing with sample questions:")
        print("-" * 40)
        
        for question in test_questions:
            print(f"\nQ: {question}")
            results = rag.search(question, n_results=2)
            
            if results:
                top_result = results[0]
                source = top_result['metadata']['source']
                similarity = top_result['similarity']
                preview = top_result['content'][:120].replace('\n', ' ')
                
                print(f"A: Found in {source} (similarity: {similarity:.3f})")
                print(f"   {preview}...")
            else:
                print("A: No relevant information found")
        
        # Run interactive demo
        rag.interactive_demo()
    else:
        print("❌ No documents loaded. Please run process_documents_simple.py first.")

if __name__ == "__main__":
    main()