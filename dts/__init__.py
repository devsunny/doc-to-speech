"""
Doc-to-Speech: A Python library for converting documents to speech.

This package provides functionality to convert various document formats
(PDF, HTML, Markdown, Text, etc.) to speech using VibeVoice TTS model.
"""

__version__ = "0.1.0"
__author__ = "devsunny"
__email__ = "your-email@example.com"

from .speaker import DocumentToSpeech
from .vibevoice_tts import TextToSpeech

__all__ = ["DocumentToSpeech", "TextToSpeech"]