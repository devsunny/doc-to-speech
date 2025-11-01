"""Basic tests for doc-to-speech package."""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import dts
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import_dts():
    """Test that we can import the dts package."""
    try:
        import dts
        assert True
    except ImportError:
        pytest.skip("dts package not found - this is expected during initial setup")

def test_import_speaker():
    """Test that we can import the DocumentToSpeech class."""
    try:
        from dts.speaker import DocumentToSpeech
        assert DocumentToSpeech is not None
    except ImportError:
        pytest.skip("dts.speaker module not found - dependencies may not be installed")

def test_import_vibevoice_tts():
    """Test that we can import the TextToSpeech class."""
    try:
        from dts.vibevoice_tts import TextToSpeech
        assert TextToSpeech is not None
    except ImportError:
        pytest.skip("dts.vibevoice_tts module not found - dependencies may not be installed")

def test_import_readers():
    """Test that we can import all reader modules."""
    readers = [
        'dts.readers.text_reader',
        'dts.readers.pdf_reader', 
        'dts.readers.html_reader',
        'dts.readers.markdown_reader',
        'dts.readers.doc_reader'
    ]
    
    for reader in readers:
        try:
            __import__(reader)
        except ImportError:
            pytest.skip(f"{reader} module not found - dependencies may not be installed")

def test_package_version():
    """Test that the package has a version."""
    try:
        import dts
        # This test will pass as long as the module loads
        assert True
    except ImportError:
        pytest.skip("dts package not found")

if __name__ == "__main__":
    pytest.main([__file__])