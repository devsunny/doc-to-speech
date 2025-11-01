

import tempfile
from pathlib import Path
from typing import Optional, Union

import torch

from dts.vibevoice_tts import TextToSpeech

MAX_CHARS = 2048  
DEFAULT_SPEAKER = "Speaker 0"

class DocumentToSpeech:
    """A class to convert document text to speech using VibeVoice TTS model."""    
     
    def __init__(self,model_id: str = "tarun7r/vibevoice-hindi-7b",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        sample_rate: int = 24000,        
        ddpm_steps: int = 5,
        cfg_scale: float = 1.3,
        lowercase: bool = False,
        sample_voice_file: Optional[Union[Path, str]] = None):        
        self.tts = TextToSpeech(
            model_id=model_id,
            device=device,
            dtype=dtype,
            sample_rate=sample_rate,
            speaker_name=DEFAULT_SPEAKER,
            ddpm_steps=ddpm_steps,
            cfg_scale=cfg_scale,
            lowercase=lowercase,
            sample_voice_file=sample_voice_file
        )

    def _read_document(self, doc_file: Union[Path, str]) -> str:
        """
        Read a document file and extract text content.
        """
        doc_ext = Path(doc_file).suffix.lower()
        if doc_ext == ".txt":
            from dts.readers.text_reader import read as text_read
            text = text_read(str(doc_file))
            return text
        elif doc_ext == ".pdf":
            from dts.readers.pdf_reader import read as pdf_read
            text = pdf_read(str(doc_file))
            if not text:
                from dts.readers.doc_reader import read as doc_read
                text = doc_read(str(doc_file))            
            return text
        elif doc_ext in [".html", ".htm"]:
            from dts.readers.html_reader import read as html_read
            text = html_read(str(doc_file))
            return text
        elif doc_ext in [".md", ".mmd"]:
            from dts.readers.markdown_reader import read as md_read
            text = md_read(str(doc_file))
            return text
        else:
            from dts.readers.doc_reader import read as doc_read
            text = doc_read(str(doc_file))
            return text    
    
    def doc_to_wav(self, doc_file: Union[Path, str], output_wav_path: Union[Path, str]=None, max_chars: int = MAX_CHARS, progress: bool = True) -> Union[None, bytes]:
        """
        Convert input text to WAV file using the TTS model.
        
        Args:
            doc_file: Path to the document file to be converted.
            output_wav_path: Path to save the output WAV file. If None, returns WAV bytes.
            max_chars: Maximum number of characters to process at once.
            progress: Whether to show progress during synthesis.
        Returns:
            None if output_wav_path is provided, else returns WAV bytes.
        """
        input_text = self._read_document(doc_file)
        if output_wav_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                output_wav_path = temp_wav.name
                self.tts.convert_text_to_wav(input_text, output_wav_path, max_chars=max_chars, progress=progress)
                return temp_wav.read()
        else:
            self.tts.convert_text_to_wav(input_text, output_wav_path, max_chars=max_chars, progress=progress)

    def doc_to_mp3(self, doc_file: Union[Path, str], output_mp3_path: Union[Path, str]=None, max_chars: int = MAX_CHARS, progress: bool = True) -> Union[None, bytes]:
        """
        Convert input text to MP3 file using the TTS model.
        
        Args:
            doc_file: Path to the document file to be converted.
            output_mp3_path: Path to save the output MP3 file. If None, returns MP3 bytes.
            max_chars: Maximum number of characters to process at once.
            progress: Whether to show progress during synthesis.
        Returns:
            None if output_mp3_path is provided, else returns MP3 bytes.
        """
        input_text = self._read_document(doc_file)
        if output_mp3_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_mp3:
                output_mp3_path = temp_mp3.name
                self.tts.convert_text_to_mp3(input_text, output_mp3_path, max_chars=max_chars, progress=progress)
                return temp_mp3.read()
        else:
            self.tts.convert_text_to_mp3(input_text, output_mp3_path, max_chars=max_chars, progress=progress)