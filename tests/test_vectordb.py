"""Tests for vector database functionality."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import json
import sys
from pathlib import Path

# Mock external dependencies
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

class MockCollection:
    """Mock ChromaDB collection."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
    def add(self, documents, embeddings, metadatas, ids):
        """Mock add method."""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embeddings, n_results=5):
        """Mock query method."""
        # Simple mock - return first n_results
        n_results = min(n_results, len(self.documents))
        
        return {
            'documents': [self.documents[:n_results]],
            'metadatas': [self.metadatas[:n_results]],
            'distances': [[0.1, 0.2, 0.3, 0.4, 0.5][:n_results]]
        }
    
    def count(self):
        """Mock count method."""
        return len(self.documents)

class MockSentenceTransformer:
    """Mock sentence transformer."""
    
    def encode(self, texts, **kwargs):
        """Mock encode method."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Return dummy embeddings
        import numpy as np
        return np.random.random((len(texts), 384))

class TestVectorDBManager:
    """Test vector database manager."""
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_init_new_collection(self, mock_st, mock_chromadb):
        """Test initialization with new collection."""
        from setup_vectordb_simple import VectorDBManager
        
        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        
        # Mock sentence transformer
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager(db_path="test_db", collection_name="test_collection")
        
        # Verify initialization
        mock_chromadb.PersistentClient.assert_called_with(path="test_db")
        mock_client.create_collection.assert_called_with(
            name="test_collection",
            metadata={"description": "Simple RAG knowledge base"}
        )
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_init_existing_collection(self, mock_st, mock_chromadb):
        """Test initialization with existing collection."""
        from setup_vectordb_simple import VectorDBManager
        
        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Make create_collection raise exception (collection exists)
        mock_client.create_collection.side_effect = Exception("Collection exists")
        mock_client.get_collection.return_value = mock_collection
        
        # Mock sentence transformer
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
        
        # Should fall back to get_collection
        mock_client.get_collection.assert_called_with(name="simple_knowledge")
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_add_documents(self, mock_st, mock_chromadb, sample_chunks):
        """Test adding documents to vector database."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            db_manager.add_documents(sample_chunks, batch_size=2)
        
        # Verify documents were added
        assert len(mock_collection.documents) == len(sample_chunks)
        assert len(mock_collection.metadatas) == len(sample_chunks)
        assert len(mock_collection.ids) == len(sample_chunks)
        
        # Check that documents match expected content
        expected_contents = [chunk['content'] for chunk in sample_chunks]
        assert mock_collection.documents == expected_contents
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer') 
    def test_search(self, mock_st, mock_chromadb, sample_chunks):
        """Test searching vector database."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        # Add sample documents first
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            db_manager.add_documents(sample_chunks, batch_size=10)
        
        # Test search
        results = db_manager.search("revenue Q3", n_results=2)
        
        # Verify search results structure
        assert isinstance(results, list)
        assert len(results) <= 2
        
        for result in results:
            assert 'content' in result
            assert 'metadata' in result
            assert 'distance' in result
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_empty_search(self, mock_st, mock_chromadb):
        """Test search with no documents in database."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks with empty collection
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
        
        # Search empty database
        results = db_manager.search("any query", n_results=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_get_stats(self, mock_st, mock_chromadb, sample_chunks):
        """Test getting database statistics."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager(db_path="test_db", collection_name="test_coll")
            db_manager.add_documents(sample_chunks)
        
        stats = db_manager.get_stats()
        
        assert 'collection_name' in stats
        assert 'total_documents' in stats
        assert 'db_path' in stats
        assert stats['collection_name'] == 'test_coll'
        assert stats['total_documents'] == len(sample_chunks)
        assert 'test_db' in stats['db_path']

class TestVectorDBEdgeCases:
    """Test edge cases for vector database."""
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_add_empty_documents(self, mock_st, mock_chromadb):
        """Test adding empty document list."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            db_manager.add_documents([], batch_size=10)
        
        assert len(mock_collection.documents) == 0
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_large_batch_processing(self, mock_st, mock_chromadb):
        """Test processing large batches of documents."""
        from setup_vectordb_simple import VectorDBManager
        
        # Create many sample chunks
        large_chunks = []
        for i in range(1000):
            large_chunks.append({
                'content': f'Document {i} content with various words and information.',
                'metadata': {'source': f'doc_{i}.txt', 'chunk_index': 0}
            })
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            db_manager.add_documents(large_chunks, batch_size=50)
        
        assert len(mock_collection.documents) == 1000
        assert len(mock_collection.ids) == 1000
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_documents_with_special_characters(self, mock_st, mock_chromadb):
        """Test handling documents with special characters."""
        from setup_vectordb_simple import VectorDBManager
        
        special_chunks = [
            {
                'content': 'Document with émojis 🚀 and spëcial chàracters',
                'metadata': {'source': 'special.txt'}
            },
            {
                'content': 'Chinese text: 这是中文内容',
                'metadata': {'source': 'chinese.txt'}
            },
            {
                'content': 'Arabic text: هذا نص عربي',
                'metadata': {'source': 'arabic.txt'}
            }
        ]
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_st.return_value = MockSentenceTransformer()
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            db_manager.add_documents(special_chunks)
        
        # Should handle special characters without issues
        assert len(mock_collection.documents) == 3
        assert '🚀' in mock_collection.documents[0]
        assert '这是中文内容' in mock_collection.documents[1]

class TestSentenceTransformerIntegration:
    """Test sentence transformer integration."""
    
    @patch('setup_vectordb_simple.chromadb')
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_sentence_transformer_initialization(self, mock_st, mock_chromadb):
        """Test that sentence transformer is initialized correctly."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
        
        # Verify SentenceTransformer was initialized with correct model
        mock_st.assert_called_with(
            'all-MiniLM-L6-v2',
            device='cpu'
        )
    
    @patch('setup_vectordb_simple.chromadb') 
    @patch('setup_vectordb_simple.SentenceTransformer')
    def test_embedding_generation(self, mock_st, mock_chromadb, sample_chunks):
        """Test that embeddings are generated for documents."""
        from setup_vectordb_simple import VectorDBManager
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MockCollection()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        
        mock_transformer = MockSentenceTransformer()
        mock_st.return_value = mock_transformer
        
        with patch('builtins.print'):
            db_manager = VectorDBManager()
            
            # Mock the encode method to track calls
            with patch.object(mock_transformer, 'encode', wraps=mock_transformer.encode) as mock_encode:
                db_manager.add_documents(sample_chunks, batch_size=10)
        
        # Verify that encode was called
        mock_encode.assert_called()
        
        # Verify that embeddings were added to collection
        assert len(mock_collection.embeddings) == len(sample_chunks)

class TestMainFunction:
    """Test the main function integration."""
    
    @patch('setup_vectordb_simple.Path')
    @patch('setup_vectordb_simple.VectorDBManager')
    def test_main_with_existing_chunks(self, mock_db_manager, mock_path):
        """Test main function with existing chunks file."""
        from setup_vectordb_simple import main
        
        # Mock chunks file exists
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Mock file content
        sample_chunks = [{'content': 'test', 'metadata': {'source': 'test.txt'}}]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_chunks))), \
             patch('json.load', return_value=sample_chunks), \
             patch('builtins.print'):
            
            main()
        
        # Verify VectorDBManager was created and documents added
        mock_db_manager.assert_called_once()
        mock_db_manager.return_value.add_documents.assert_called_with(sample_chunks, batch_size=100)
    
    @patch('setup_vectordb_simple.Path')
    def test_main_with_missing_chunks(self, mock_path):
        """Test main function with missing chunks file.""" 
        from setup_vectordb_simple import main
        
        # Mock chunks file doesn't exist
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            
            main()
        
        # Should print error message and exit
        mock_print.assert_called()
        mock_exit.assert_called_with(1)