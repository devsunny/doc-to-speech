import os
from pathlib import Path
import re
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Iterable, List, Optional, Union
from pydub import AudioSegment

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


class TextToSpeech:
    """
    Long-form Text-to-Speech using VibeVoice inference, with:
      - text cleaning (removes punctuation, normalizes whitespace)
      - chunking for long text
      - single model/processor load for efficiency
    """

    def __init__(
        self,
        model_id: str = "tarun7r/vibevoice-hindi-7b",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        sample_rate: int = 24000,
        speaker_name: str = "Speaker 0",
        ddpm_steps: int = 5,
        cfg_scale: float = 1.3,
        lowercase: bool = False,
        sample_voice_file: Optional[Union[Path, str]] = None,
    ):
        """
        Args:
            model_id: HF repo or local path to VibeVoice model.
            device: "cuda", "mps", or "cpu". Auto-detected if None.
            dtype: torch.float16 for CUDA, torch.float32 otherwise if None.
            sample_rate: output sample rate (VibeVoice commonly uses 24k).
            speaker_name: label required by VibeVoice ("Speaker 0:", "Alice:", etc.)
            ddpm_steps: inference steps for diffusion part.
            cfg_scale: guidance scale during generation.
            lowercase: optionally lowercase the input text during cleaning.
        """
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.speaker_name = speaker_name
        self.cfg_scale = cfg_scale
        self.lowercase = lowercase

        # Device & dtype defaults
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype

        # Load processor & model once (efficient)
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_id)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.model.set_ddpm_inference_steps(ddpm_steps)

        # Cache for voice samples once prepared
        self._voice_samples: Optional[List[List[np.ndarray]]] = None
        if sample_voice_file is None:
            sample_voice_file = Path(__file__).parent / "default_voice_sample.wav"
            
        if sample_voice_file is not None:
            self.load_voice_sample(voice_path=str(sample_voice_file))

    # ------------------------- audio utilities -------------------------

    @staticmethod
    def _to_mono_24k(audio: np.ndarray, sr: int, target_sr: int = 24000) -> np.ndarray:
        """Ensure mono and 24kHz float32."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)

    @staticmethod
    def _safe_audio_buffer(x: np.ndarray) -> np.ndarray:
        """Clamp, de-NaN, and ensure float32 1-D."""
        x = np.squeeze(x)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        return x

    # ------------------------- text utilities -------------------------

    def _clean_text(self, text: str) -> str:
        """
        Remove punctuation and normalize whitespace.
        If lowercase=True, also make lowercase.
        """
        if self.lowercase:
            text = text.lower()
        # remove punctuation (keep letters, numbers, whitespace)
        # you can tweak allowed chars if needed (e.g., keep apostrophes)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
        return text

    @staticmethod
    def _split_long_text(text: str, max_chars: int = 350) -> List[str]:
        """
        Split long text into word-boundary chunks <= max_chars.
        Tune max_chars to your GPU/VRAM & stability needs.
        """
        words = text.split()
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for w in words:
            wlen = len(w) + (1 if cur_len > 0 else 0)
            if cur_len + wlen > max_chars:
                if cur:
                    chunks.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += wlen
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def _format_as_speaker_lines(self, chunks: Iterable[str]) -> List[str]:
        """
        VibeVoice requires speaker-labeled lines: "<Name>: text"
        """
        return [f"{self.speaker_name}: {c}" for c in chunks]

    # ------------------------- public API -------------------------

    def load_voice_sample(
        self,
        voice_path: Optional[str] = None,
        voice_array: Optional[np.ndarray] = None,
        voice_sr: Optional[int] = None,
    ) -> None:
        """
        Prepare (mono, 24k) voice sample(s) and cache for inference.
        Provide either `voice_path` OR (`voice_array` + `voice_sr`).
        """
        if voice_path:
            audio, sr = sf.read(voice_path)
        elif voice_array is not None and voice_sr is not None:
            audio, sr = voice_array, voice_sr
        else:
            raise ValueError("Provide either voice_path or (voice_array, voice_sr)")

        audio = self._to_mono_24k(np.asarray(audio), sr, self.sample_rate)
        # VibeVoice expects List[List[np.ndarray]] (per speaker → list of refs)
        self._voice_samples = [[audio]]

    def synthesize_chunk(self, text_chunk: str) -> np.ndarray:
        """
        Synthesize a single cleaned chunk and return audio samples (float32 24k mono).
        """
        if self._voice_samples is None:
            raise RuntimeError("Call load_voice_sample(...) before synthesize_chunk.")

        # Clean & format as speaker labeled line
        cleaned = self._clean_text(text_chunk)
        line = f"{self.speaker_name}: {cleaned}"

        inputs = self.processor(
            text=[line],                       # list of lines
            voice_samples=self._voice_samples, # [[numpy-array(s) for this speaker]]
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to model device (only tensor entries)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            if self.device == "cuda" and self.dtype == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.generate(**inputs, cfg_scale=self.cfg_scale, tokenizer=self.processor.tokenizer)
            else:
                out = self.model.generate(**inputs, cfg_scale=self.cfg_scale, tokenizer=self.processor.tokenizer)

        audio = out.speech_outputs[0].detach().cpu().float().numpy()
        return self._safe_audio_buffer(audio)

    def synthesize_text(
        self,
        text: str,
        max_chars: int = 350,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Synthesize long text by chunking -> per-chunk inference -> concat.
        Returns a mono float32 array at self.sample_rate.
        """
        cleaned = self._clean_text(text)
        chunks = self._split_long_text(cleaned, max_chars=max_chars)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        all_audio: List[np.ndarray] = []
        total = len(chunks)
        for i, ch in enumerate(chunks, 1):
            if progress:
                print(f"[TTS] Generating chunk {i}/{total} (len={len(ch)})...")
            audio = self.synthesize_chunk(ch)
            all_audio.append(audio)

        if len(all_audio) == 1:
            return all_audio[0]
        return self._safe_audio_buffer(np.concatenate(all_audio, axis=0))

    def save_wav(self, path: str, samples: np.ndarray) -> None:
        """Write WAV as PCM_16 (robust on Windows)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        x = self._safe_audio_buffer(samples)
        sf.write(path, x, self.sample_rate, format="WAV", subtype="PCM_16")
        
    
    def convert_text_to_wav(self, input_text: str, output_wav_path: str, max_chars: int = 350, progress: bool = True) -> None:
        """
        Convenience method to synthesize text and save directly to WAV file.
        """
        audio = self.synthesize_text(input_text, max_chars=max_chars, progress=progress)
        self.save_wav(output_wav_path, audio)
        
    
    def convert_text_to_mp3(self, input_text: str, output_wav_path: str, max_chars: int = 350, progress: bool = True) -> None:
        """
        Convenience method to synthesize text and save directly to WAV file.
        """
        audio = self.synthesize_text(input_text, max_chars=max_chars, progress=progress)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            self.save_wav(temp_wav.name, audio)
            sound = AudioSegment.from_wav(temp_wav.name)
            sound.export(output_wav_path, format="mp3")
    