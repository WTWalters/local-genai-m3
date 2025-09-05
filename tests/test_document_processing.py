"""Tests for document processing functionality."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# Mock docling import since it might not be available in test environment
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()

class TestDocumentProcessing:
    """Test document processing functionality."""
    
    def test_process_markdown_file(self, sample_documents):
        """Test processing a markdown file."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(sample_documents))
        md_file = sample_documents / "sample.md"
        
        result = processor.process_document(md_file)
        
        assert result['status'] == 'success'
        assert 'Q3 2024 Report' in result['content']
        assert result['metadata']['source'] == 'sample.md'
        assert result['metadata']['format'] == '.md'
    
    def test_process_text_file(self, sample_documents):
        """Test processing a text file."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(sample_documents))
        txt_file = sample_documents / "risks.txt"
        
        result = processor.process_document(txt_file)
        
        assert result['status'] == 'success'
        assert 'Strategic Risk Assessment' in result['content']
        assert result['metadata']['source'] == 'risks.txt'
        assert result['metadata']['format'] == '.txt'
    
    def test_process_nonexistent_file(self, temp_dir):
        """Test processing a file that doesn't exist."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        nonexistent = temp_dir / "missing.md"
        
        with patch('builtins.print'):
            result = processor.process_document(nonexistent)
        
        assert result['status'] == 'error'
        assert 'error' in result
        assert result['content'] is None
    
    def test_chunk_text_basic(self, temp_dir):
        """Test basic text chunking."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        # Test with short text
        short_text = "This is a short text that should be one chunk."
        chunks = processor.chunk_text(short_text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Test with longer text
        long_text = "This is a longer piece of text. " * 20
        chunks = processor.chunk_text(long_text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50 + 20  # Allow for word boundaries
    
    def test_chunk_text_with_overlap(self, temp_dir):
        """Test text chunking with overlap."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = processor.chunk_text(text, chunk_size=30, overlap=10)
        
        # Should have overlap between chunks
        assert len(chunks) >= 2
        
        # Check for overlap (this is a simplified check)
        if len(chunks) > 1:
            # There should be some common words between adjacent chunks
            words1 = set(chunks[0].split())
            words2 = set(chunks[1].split())
            assert len(words1.intersection(words2)) > 0
    
    def test_process_corpus_empty_directory(self, temp_dir):
        """Test processing an empty directory.""" 
        from process_documents_simple import DocumentProcessor
        
        empty_dir = temp_dir / "empty_docs"
        empty_dir.mkdir()
        
        processor = DocumentProcessor(str(empty_dir))
        
        with patch('builtins.print'):
            chunks = processor.process_corpus()
        
        assert chunks == []
    
    def test_process_corpus_with_documents(self, sample_documents):
        """Test processing a directory with documents."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(sample_documents))
        
        with patch('builtins.print'):
            chunks = processor.process_corpus()
        
        assert len(chunks) > 0
        
        # Check chunk structure
        for chunk in chunks:
            assert 'content' in chunk
            assert 'metadata' in chunk
            assert 'source' in chunk['metadata']
            assert 'chunk_index' in chunk['metadata']
            assert 'total_chunks' in chunk['metadata']
    
    def test_process_corpus_saves_output(self, sample_documents, temp_dir):
        """Test that corpus processing saves output file."""
        from process_documents_simple import DocumentProcessor
        
        # Mock the output path
        with patch.object(Path, 'mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('builtins.print'):
            
            processor = DocumentProcessor(str(sample_documents))
            chunks = processor.process_corpus()
            
            # Should have called json.dump to save the results
            mock_json_dump.assert_called_once()

class TestDocumentProcessorEdgeCases:
    """Test edge cases in document processing."""
    
    def test_empty_file_processing(self, temp_dir):
        """Test processing an empty file."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        
        result = processor.process_document(empty_file)
        
        assert result['status'] == 'success'
        assert result['content'] == ''
    
    def test_very_large_file(self, temp_dir):
        """Test processing a very large file."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        # Create a large file
        large_file = temp_dir / "large.txt"
        large_content = "This is a line of text. " * 10000
        large_file.write_text(large_content)
        
        result = processor.process_document(large_file)
        
        assert result['status'] == 'success'
        assert len(result['content']) > 100000
    
    def test_file_with_special_characters(self, temp_dir):
        """Test processing file with special characters."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        special_file = temp_dir / "special.txt"
        special_content = "Special chars: àáâãäå çñü €$¥ 中文 العربية 🚀🎉"
        special_file.write_text(special_content, encoding='utf-8')
        
        result = processor.process_document(special_file)
        
        assert result['status'] == 'success'
        assert 'Special chars:' in result['content']
        assert '🚀' in result['content']
    
    def test_unsupported_file_types(self, temp_dir):
        """Test handling of unsupported file types."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        # Create files with various extensions
        binary_file = temp_dir / "image.jpg"
        binary_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')
        
        code_file = temp_dir / "script.py"
        code_file.write_text("print('hello world')")
        
        with patch('builtins.print'):
            chunks = processor.process_corpus()
        
        # Should not process unsupported file types
        processed_sources = [chunk['metadata']['source'] for chunk in chunks]
        assert 'image.jpg' not in processed_sources
        # Python files might be processed depending on implementation
    
    def test_permission_denied_file(self, temp_dir):
        """Test handling of files with permission issues."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        # Create a file and remove read permissions
        protected_file = temp_dir / "protected.txt"
        protected_file.write_text("Protected content")
        protected_file.chmod(0o000)
        
        try:
            result = processor.process_document(protected_file)
            assert result['status'] == 'error'
        except PermissionError:
            # This is also acceptable behavior
            pass
        finally:
            # Restore permissions for cleanup
            try:
                protected_file.chmod(0o644)
            except:
                pass

class TestChunkingStrategies:
    """Test different chunking strategies."""
    
    def test_sentence_boundary_chunking(self, temp_dir):
        """Test that chunking respects sentence boundaries."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        text = ("First sentence. Second sentence. Third sentence. "
                "Fourth sentence. Fifth sentence. Sixth sentence.")
        
        chunks = processor.chunk_text(text, chunk_size=40, overlap=10)
        
        # Chunks should generally end at sentence boundaries
        for chunk in chunks:
            if not chunk.endswith('.'):
                # If not ending with period, should be end of text or reasonable break
                assert chunk == chunks[-1] or '.' in chunk
    
    def test_word_boundary_chunking(self, temp_dir):
        """Test that chunking respects word boundaries."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        text = "This is a very long word: supercalifragilisticexpialidocious and more text here."
        
        chunks = processor.chunk_text(text, chunk_size=20, overlap=5)
        
        # Chunks should not split words
        for chunk in chunks:
            words = chunk.split()
            if words:
                # Each word should be complete (no partial words at boundaries)
                # This is a simplified check
                assert not chunk.startswith(' ')
                assert not chunk.endswith(' ') or chunk == chunks[-1]
    
    def test_minimum_chunk_size(self, temp_dir):
        """Test minimum chunk size enforcement."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(temp_dir))
        
        # Very short text
        short_text = "Hi."
        chunks = processor.chunk_text(short_text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Text with many tiny fragments
        fragments = "A. B. C. D. E. F. G. H. I. J."
        chunks = processor.chunk_text(fragments, chunk_size=5, overlap=1)
        
        # Should handle gracefully
        assert len(chunks) > 0
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

class TestMetadataHandling:
    """Test metadata extraction and handling."""
    
    def test_metadata_extraction(self, sample_documents):
        """Test that metadata is correctly extracted."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(sample_documents))
        md_file = sample_documents / "sample.md"
        
        result = processor.process_document(md_file)
        metadata = result['metadata']
        
        assert metadata['source'] == 'sample.md'
        assert metadata['path'] == str(md_file)
        assert metadata['format'] == '.md'
    
    def test_chunk_metadata_inheritance(self, sample_documents):
        """Test that chunks inherit and extend document metadata."""
        from process_documents_simple import DocumentProcessor
        
        processor = DocumentProcessor(str(sample_documents))
        
        with patch('builtins.print'):
            chunks = processor.process_corpus()
        
        for chunk in chunks:
            metadata = chunk['metadata']
            
            # Should have document-level metadata
            assert 'source' in metadata
            assert 'format' in metadata
            
            # Should have chunk-specific metadata
            assert 'chunk_index' in metadata
            assert 'total_chunks' in metadata
            assert isinstance(metadata['chunk_index'], int)
            assert isinstance(metadata['total_chunks'], int)
            assert metadata['chunk_index'] >= 0
            assert metadata['total_chunks'] > 0
            assert metadata['chunk_index'] < metadata['total_chunks']