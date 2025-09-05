"""Utility tests and helper functions."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

class TestUtilities:
    """Test utility functions and helpers."""
    
    def test_environment_setup(self, mock_env_vars):
        """Test environment variable setup."""
        
        # Check that environment variables are properly set
        assert os.environ.get('PROJECT_ROOT') is not None
        assert os.environ.get('MODEL_PATH') is not None
        assert os.environ.get('DATA_PATH') is not None
        assert os.environ.get('CHECKPOINT_PATH') is not None
        
        # Check that directories exist
        assert Path(os.environ['MODEL_PATH']).exists()
        assert Path(os.environ['DATA_PATH']).exists()
        assert Path(os.environ['CHECKPOINT_PATH']).exists()
    
    def test_temp_directory_fixture(self, temp_dir):
        """Test temp directory fixture."""
        
        # Should be a Path object
        assert isinstance(temp_dir, Path)
        
        # Should exist and be writable
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Should be able to create files
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_sample_chunks_fixture(self, sample_chunks):
        """Test sample chunks fixture."""
        
        # Should be a list
        assert isinstance(sample_chunks, list)
        assert len(sample_chunks) > 0
        
        # Each chunk should have required fields
        for chunk in sample_chunks:
            assert 'content' in chunk
            assert 'metadata' in chunk
            assert isinstance(chunk['content'], str)
            assert isinstance(chunk['metadata'], dict)
            
            # Metadata should have required fields
            assert 'source' in chunk['metadata']
            assert 'chunk_index' in chunk['metadata']
    
    def test_sample_documents_fixture(self, sample_documents):
        """Test sample documents fixture."""
        
        # Should be a Path to an existing directory
        assert isinstance(sample_documents, Path)
        assert sample_documents.exists()
        assert sample_documents.is_dir()
        
        # Should contain sample files
        files = list(sample_documents.glob("*"))
        assert len(files) > 0
        
        # Files should be readable
        for file in files:
            if file.suffix in ['.md', '.txt']:
                content = file.read_text()
                assert len(content) > 0
    
    def test_sample_chunks_file_fixture(self, sample_chunks_file, sample_chunks):
        """Test sample chunks file fixture."""
        
        # Should be a Path to existing file
        assert isinstance(sample_chunks_file, Path)
        assert sample_chunks_file.exists()
        assert sample_chunks_file.suffix == '.json'
        
        # Should contain valid JSON
        with open(sample_chunks_file, 'r') as f:
            loaded_chunks = json.load(f)
        
        assert isinstance(loaded_chunks, list)
        assert len(loaded_chunks) == len(sample_chunks)
        
        # Content should match original chunks
        for original, loaded in zip(sample_chunks, loaded_chunks):
            assert original['content'] == loaded['content']
            assert original['metadata'] == loaded['metadata']

class TestFileOperations:
    """Test file operation utilities."""
    
    def test_safe_file_creation(self, temp_dir):
        """Test safe file creation patterns."""
        
        # Test creating file in subdirectory
        subdir = temp_dir / "subdir" 
        subdir.mkdir()
        
        test_file = subdir / "test.json"
        test_data = {"test": "data"}
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        assert test_file.exists()
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
    
    def test_file_cleanup_patterns(self, temp_dir):
        """Test file cleanup patterns."""
        
        # Create various test files
        files_to_create = [
            "test.txt",
            "data.json", 
            "model.bin",
            "checkpoint.pth"
        ]
        
        created_files = []
        for filename in files_to_create:
            file_path = temp_dir / filename
            file_path.write_text(f"content for {filename}")
            created_files.append(file_path)
            assert file_path.exists()
        
        # Files should be automatically cleaned up when temp_dir is destroyed
        # This is handled by the fixture cleanup
    
    def test_large_file_handling(self, temp_dir):
        """Test handling of larger files."""
        
        # Create a moderately large file
        large_file = temp_dir / "large.txt"
        content = "Line of text.\n" * 10000  # ~140KB
        
        large_file.write_text(content)
        
        # Should be able to read it back
        read_content = large_file.read_text()
        assert len(read_content) == len(content)
        assert read_content == content
        
        # Check file size
        stat = large_file.stat()
        assert stat.st_size > 100000  # At least 100KB

class TestErrorHandlingPatterns:
    """Test common error handling patterns."""
    
    def test_missing_file_handling(self, temp_dir):
        """Test handling of missing files."""
        
        missing_file = temp_dir / "missing.json"
        
        # Should not exist
        assert not missing_file.exists()
        
        # Reading should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            missing_file.read_text()
        
        # JSON loading should raise appropriate exception
        with pytest.raises(FileNotFoundError):
            with open(missing_file, 'r') as f:
                json.load(f)
    
    def test_invalid_json_handling(self, temp_dir):
        """Test handling of invalid JSON."""
        
        invalid_json_file = temp_dir / "invalid.json"
        invalid_json_file.write_text("{ invalid json content")
        
        # Should raise JSON decode error
        with pytest.raises(json.JSONDecodeError):
            with open(invalid_json_file, 'r') as f:
                json.load(f)
    
    def test_permission_error_simulation(self, temp_dir):
        """Test permission error handling."""
        
        # Create file and make it unreadable (on Unix systems)
        if os.name == 'posix':  # Unix-like systems
            protected_file = temp_dir / "protected.txt"
            protected_file.write_text("protected content")
            
            # Remove read permissions
            protected_file.chmod(0o000)
            
            try:
                # Should raise PermissionError
                with pytest.raises(PermissionError):
                    protected_file.read_text()
            finally:
                # Restore permissions for cleanup
                try:
                    protected_file.chmod(0o644)
                except:
                    pass

class TestDataValidation:
    """Test data validation patterns."""
    
    def test_chunk_data_validation(self, sample_chunks):
        """Test validation of chunk data structure."""
        
        for chunk in sample_chunks:
            # Required fields
            assert 'content' in chunk
            assert 'metadata' in chunk
            
            # Type validation
            assert isinstance(chunk['content'], str)
            assert isinstance(chunk['metadata'], dict)
            
            # Content validation
            assert len(chunk['content']) > 0
            assert len(chunk['metadata']) > 0
            
            # Metadata validation
            metadata = chunk['metadata']
            assert 'source' in metadata
            assert isinstance(metadata['source'], str)
            assert len(metadata['source']) > 0
    
    def test_metadata_structure_validation(self, sample_chunks):
        """Test metadata structure validation."""
        
        for chunk in sample_chunks:
            metadata = chunk['metadata']
            
            # Required fields for chunk metadata
            required_fields = ['source', 'chunk_index']
            for field in required_fields:
                assert field in metadata, f"Missing required field: {field}"
            
            # Type validation
            assert isinstance(metadata['chunk_index'], int)
            assert metadata['chunk_index'] >= 0
            
            # Optional fields validation
            if 'total_chunks' in metadata:
                assert isinstance(metadata['total_chunks'], int)
                assert metadata['total_chunks'] > 0
                assert metadata['chunk_index'] < metadata['total_chunks']
    
    def test_empty_data_handling(self, temp_dir):
        """Test handling of empty data structures."""
        
        # Empty chunks list
        empty_chunks_file = temp_dir / "empty_chunks.json"
        with open(empty_chunks_file, 'w') as f:
            json.dump([], f)
        
        # Should load without error
        with open(empty_chunks_file, 'r') as f:
            empty_chunks = json.load(f)
        
        assert isinstance(empty_chunks, list)
        assert len(empty_chunks) == 0
        
        # Empty chunk content
        empty_content_chunk = {
            'content': '',
            'metadata': {'source': 'empty.txt', 'chunk_index': 0}
        }
        
        # Should be valid structure even with empty content
        assert 'content' in empty_content_chunk
        assert isinstance(empty_content_chunk['content'], str)

class TestPerformancePatterns:
    """Test performance-related patterns.""" 
    
    def test_large_list_processing(self):
        """Test processing of large lists."""
        
        # Create large list
        large_list = list(range(10000))
        
        # Should process without issues
        processed = [x * 2 for x in large_list]
        
        assert len(processed) == len(large_list)
        assert processed[0] == 0
        assert processed[-1] == 19998
    
    def test_memory_efficient_iteration(self, temp_dir):
        """Test memory-efficient iteration patterns."""
        
        # Create file with many lines
        many_lines_file = temp_dir / "many_lines.txt"
        
        lines = [f"Line {i}\n" for i in range(1000)]
        many_lines_file.write_text(''.join(lines))
        
        # Test line-by-line reading (memory efficient)
        line_count = 0
        with open(many_lines_file, 'r') as f:
            for line in f:
                line_count += 1
                assert line.startswith("Line ")
        
        assert line_count == 1000
    
    @pytest.mark.slow
    def test_cpu_intensive_operation(self):
        """Test CPU-intensive operations (marked as slow)."""
        
        # Simple CPU-intensive task
        result = sum(i * i for i in range(10000))
        
        # Mathematical verification
        expected = (10000 * (10000 - 1) * (2 * 10000 - 1)) // 6
        assert result == expected

class TestMockingPatterns:
    """Test common mocking patterns used in tests."""
    
    def test_print_mocking_pattern(self):
        """Demonstrate print mocking pattern."""
        
        from unittest.mock import patch
        with patch('builtins.print') as mock_print:
            print("This should be mocked")
            print("Multiple calls")
        
        # Verify print was called
        assert mock_print.call_count == 2
        mock_print.assert_called_with("Multiple calls")
    
    def test_file_mocking_pattern(self):
        """Demonstrate file mocking pattern."""
        
        from unittest.mock import patch, mock_open
        mock_content = "mocked file content"
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            with open('any_file.txt', 'r') as f:
                content = f.read()
        
        assert content == mock_content
    
    def test_environment_variable_mocking(self):
        """Demonstrate environment variable mocking."""
        
        from unittest.mock import patch
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            assert os.environ.get('TEST_VAR') == 'test_value'
        
        # Should not exist outside the patch
        assert os.environ.get('TEST_VAR') is None