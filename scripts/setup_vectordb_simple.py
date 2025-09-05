#!/usr/bin/env python3
"""Setup ChromaDB vector database with simple embeddings."""

import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class SimpleVectorDB:
    """Simple vector database manager using ChromaDB."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "business_knowledge"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        print("Initializing vector database...")
        
        # Load embedding model (using a simpler model for compatibility)
        print("Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",  # Smaller, more compatible model
                device="cpu"  # Use CPU for compatibility
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            print("Trying alternative model...")
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            print(f"✅ ChromaDB initialized at: {self.db_path}")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            raise
        
        # Create or get collection
        try:
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Using existing collection: {self.collection_name}")
                print(f"   Existing documents: {self.collection.count()}")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Business knowledge base for RAG"}
                )
                print(f"✅ Created new collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Collection setup failed: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            try:
                self.client.delete_collection(name=self.collection_name)
                print("✅ Cleared existing collection")
            except:
                pass  # Collection might not exist
            
            # Create fresh collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Business knowledge base for RAG"}
            )
            print("✅ Created fresh collection")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
    
    def add_documents(self, chunks: list, batch_size: int = 32, clear_first: bool = True):
        """Add document chunks to the vector database."""
        
        if clear_first:
            self.clear_collection()
        
        print(f"Adding {len(chunks)} chunks to vector database...")
        
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
            
            try:
                # Generate embeddings
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=min(batch_size, 16)  # Smaller batch for stability
                )
                
                # Add to collection
                self.collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
            except Exception as e:
                print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        final_count = self.collection.count()
        print(f"✅ Added chunks to vector database")
        print(f"   Total documents in collection: {final_count}")
        
        return final_count
    
    def search(self, query: str, n_results: int = 5):
        """Search the vector database."""
        
        print(f"Searching for: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
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
            
            print(f"✅ Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self):
        """Get database statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "db_path": str(self.db_path),
                "embedding_model": self.embedding_model.get_model_card_data().model_id if hasattr(self.embedding_model, 'get_model_card_data') else "unknown"
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
    
    def test_search_functionality(self):
        """Test search with sample queries."""
        
        print("\n" + "="*60)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*60)
        
        test_queries = [
            "What is the Q3 revenue?",
            "Tell me about Project Phoenix",
            "What are the main strategic risks?",
            "How much did we spend on technology infrastructure?",
            "What is our customer acquisition rate?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            
            results = self.search(query, n_results=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    source = result['metadata'].get('source', 'Unknown')
                    distance = result.get('distance', 'N/A')
                    content_preview = result['content'][:150].replace('\n', ' ')
                    
                    print(f"{i}. Source: {source}")
                    print(f"   Distance: {distance}")
                    print(f"   Content: {content_preview}...")
                    print()
            else:
                print("   No results found")

def main():
    """Main function to setup and test vector database."""
    
    # Load processed chunks
    chunks_path = Path("data/processed_chunks.json")
    
    if not chunks_path.exists():
        print("❌ No processed chunks found. Run process_documents_simple.py first.")
        return
    
    print("Loading processed chunks...")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Setup vector database
    try:
        db_manager = SimpleVectorDB()
        
        # Add documents
        final_count = db_manager.add_documents(chunks, batch_size=16)
        
        if final_count > 0:
            # Show stats
            stats = db_manager.get_stats()
            print(f"\nDatabase Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Test search functionality
            db_manager.test_search_functionality()
        else:
            print("❌ No documents were added to the database")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()