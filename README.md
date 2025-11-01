# Doc-to-Speech

A Python library and CLI tool for converting various document formats to speech using VibeVoice TTS (Text-to-Speech) model.

## Features

- **Multiple Document Format Support**: Converts text, PDF, HTML, Markdown, and other document formats to speech
- **High-Quality TTS**: Uses VibeVoice Hindi 7B model for natural-sounding speech synthesis
- **Multiple Output Formats**: Supports both WAV and MP3 audio output
- **Smart Text Processing**: Handles long documents by intelligently chunking text while preserving word boundaries
- **Configurable Voice**: Supports custom voice samples for personalized speech synthesis
- **GPU Acceleration**: Optimized for CUDA, MPS, and CPU execution
- **Command Line Interface**: Easy-to-use CLI for batch processing and automation

## Supported Document Formats

- **Text Files** (`.txt`)
- **PDF Documents** (`.pdf`) - Using PyMuPDF
- **HTML Files** (`.html`, `.htm`) - Extracts clean text content
- **Markdown Files** (`.md`, `.mmd`) - Converts to plain text
- **Office Documents** - Word, PowerPoint, Excel and other formats via Docling

## Installation

```bash
pip install doc-to-speech
```

### Requirements

- Python >= 3.12
- PyTorch with CUDA support (recommended for best performance)
- See `pyproject.toml` for complete dependency list

## Quick Start

### CLI Usage (Recommended for most users)

```bash
# Convert a document to WAV
dts wav --source document.pdf --output speech.wav

# Convert a document to MP3  
dts mp3 --source document.txt --output speech.mp3

# Convert WAV to MP3
dts wav2mp3 --source audio.wav --output compressed.mp3

# Get help
dts --help
```

### Python API Usage

```python
from dts.speaker import DocumentToSpeech

# Initialize the TTS engine
tts = DocumentToSpeech()

# Convert a document to WAV audio
tts.doc_to_wav("document.pdf", "output.wav")

# Convert a document to MP3 audio
tts.doc_to_mp3("document.txt", "output.mp3")
```

## Command Line Interface

After installation, you can use the CLI commands for quick conversions:

### Available Commands

```bash
# Convert documents to WAV format
dts wav --source document.pdf --output speech.wav
dts wav -s folder_with_docs/ -o output.wav

# Convert documents to MP3 format  
dts mp3 --source document.txt --output speech.mp3
dts mp3 -s folder_with_docs/ -o output.mp3

# Convert existing WAV files to MP3
dts wav2mp3 --source audio.wav --output compressed.mp3
dts wav2mp3 -s audio_folder/ -o output.mp3
```

### CLI Examples

#### Single File Conversion
```bash
# Convert a PDF to speech
dts wav -s report.pdf -o report_audio.wav

# Convert a text file to MP3
dts mp3 -s notes.txt -o notes_audio.mp3
```

#### Batch Processing
```bash
# Convert all documents in a folder to WAV
dts wav -s documents/ -o batch_output.wav

# Convert all documents in a folder to MP3
dts mp3 -s documents/ -o batch_output.mp3

# Convert all WAV files in a folder to MP3
dts wav2mp3 -s audio_files/ -o compressed_output.mp3
```

#### Getting Help
```bash
# Show general help
dts --help

# Show help for specific commands
dts wav --help
dts mp3 --help
dts wav2mp3 --help
```

Alternative command name:
```bash
# You can also use the full name
doc-to-speech wav -s document.pdf -o output.wav
```

### Advanced Configuration

```python
from dts.speaker import DocumentToSpeech
import torch

# Custom configuration
tts = DocumentToSpeech(
    model_id="tarun7r/vibevoice-hindi-7b",  # TTS model
    device="cuda",                          # Device: "cuda", "mps", or "cpu"
    dtype=torch.float16,                    # Data type for optimization
    sample_rate=24000,                      # Audio sample rate
    ddpm_steps=5,                          # Diffusion steps
    cfg_scale=1.3,                         # Guidance scale
    lowercase=False,                       # Text preprocessing
    sample_voice_file="voice_sample.wav"   # Custom voice sample
)

# Process with custom settings
tts.doc_to_wav(
    doc_file="long_document.pdf",
    output_wav_path="speech.wav",
    max_chars=2048,                        # Characters per chunk
    progress=True                          # Show progress
)
```

### Using Custom Voice Samples

```python
# Load a custom voice sample
tts = DocumentToSpeech(sample_voice_file="my_voice.wav")

# Or load programmatically
from dts.vibevoice_tts import TextToSpeech
tts_engine = TextToSpeech()
tts_engine.load_voice_sample(voice_path="custom_voice.wav")
```

## API Reference

### DocumentToSpeech Class

The main class for document-to-speech conversion.

#### Constructor Parameters

- `model_id` (str): HuggingFace model ID (default: "tarun7r/vibevoice-hindi-7b")
- `device` (str, optional): Computing device ("cuda", "mps", "cpu")
- `dtype` (torch.dtype, optional): Data type for model computation
- `sample_rate` (int): Audio sample rate (default: 24000)
- `ddpm_steps` (int): Diffusion model steps (default: 5)
- `cfg_scale` (float): Classifier-free guidance scale (default: 1.3)
- `lowercase` (bool): Convert text to lowercase (default: False)
- `sample_voice_file` (str, optional): Path to voice sample file

#### Methods

##### `doc_to_wav(doc_file, output_wav_path=None, max_chars=2048, progress=True)`

Convert document to WAV audio format.

**Parameters:**
- `doc_file` (str|Path): Path to input document
- `output_wav_path` (str|Path, optional): Output WAV file path
- `max_chars` (int): Maximum characters per processing chunk
- `progress` (bool): Show progress during conversion

**Returns:** WAV bytes if `output_wav_path` is None, otherwise None

##### `doc_to_mp3(doc_file, output_mp3_path=None, max_chars=2048, progress=True)`

Convert document to MP3 audio format.

**Parameters:**
- `doc_file` (str|Path): Path to input document  
- `output_mp3_path` (str|Path, optional): Output MP3 file path
- `max_chars` (int): Maximum characters per processing chunk
- `progress` (bool): Show progress during conversion

**Returns:** MP3 bytes if `output_mp3_path` is None, otherwise None

### TextToSpeech Class

Low-level TTS engine for direct text synthesis.

#### Key Methods

##### `synthesize_text(text, max_chars=350, progress=True)`

Synthesize audio from text string.

##### `convert_text_to_wav(input_text, output_wav_path, max_chars=350, progress=True)`

Convert text directly to WAV file.

##### `convert_text_to_mp3(input_text, output_mp3_path, max_chars=350, progress=True)`

Convert text directly to MP3 file.

##### `load_voice_sample(voice_path=None, voice_array=None, voice_sr=None)`

Load custom voice sample for personalized synthesis.

## Document Readers

The library includes specialized readers for different document formats:

### Text Reader (`dts.readers.text_reader`)
- Handles plain text files with UTF-8 encoding

### PDF Reader (`dts.readers.pdf_reader`)  
- Uses PyMuPDF for robust PDF text extraction
- Preserves text structure and formatting

### HTML Reader (`dts.readers.html_reader`)
- Uses BeautifulSoup for clean text extraction
- Removes scripts, styles, and other non-content elements
- Normalizes whitespace

### Markdown Reader (`dts.readers.markdown_reader`)
- Converts Markdown to HTML then to plain text
- Preserves document structure while removing formatting

### Document Reader (`dts.readers.doc_reader`)
- Uses Docling for comprehensive document format support
- Handles Word, PowerPoint, Excel, and other office formats

## Performance Optimization

### GPU Acceleration

The library automatically detects and uses available GPU acceleration:

```python
# CUDA (NVIDIA GPUs)
tts = DocumentToSpeech(device="cuda", dtype=torch.float16)

# MPS (Apple Silicon)
tts = DocumentToSpeech(device="mps")

# CPU (fallback)
tts = DocumentToSpeech(device="cpu", dtype=torch.float32)
```

### Memory Management

For large documents, adjust the `max_chars` parameter to balance quality and memory usage:

```python
# Conservative setting for limited memory
tts.doc_to_wav("large_doc.pdf", "output.wav", max_chars=350)

# Higher throughput for capable systems
tts.doc_to_wav("large_doc.pdf", "output.wav", max_chars=2048)
```

## Text Processing

The library includes intelligent text processing:

- **Punctuation Removal**: Cleans text for better TTS quality
- **Whitespace Normalization**: Handles inconsistent formatting
- **Smart Chunking**: Splits long text at word boundaries
- **Speaker Formatting**: Formats text for VibeVoice model requirements

## Error Handling

The library gracefully handles common issues:

- Falls back between different PDF reading methods
- Provides informative error messages for unsupported formats
- Validates audio processing parameters
- Handles memory constraints through chunking

## Examples

### CLI Examples

#### Convert Multiple Documents with CLI
```bash
# Convert all PDFs in a directory to WAV
dts wav -s documents/ -o audio/

# Convert specific file types to MP3
find documents/ -name "*.pdf" -exec dts mp3 -s {} -o audio/{}.mp3 \;

# Convert and compress existing audio
dts wav2mp3 -s generated_audio/ -o compressed/
```

#### Automated Batch Processing
```bash
#!/bin/bash
# Script to process different document types

# Process all text files
for file in documents/*.txt; do
    dts wav -s "$file" -o "audio/$(basename "$file" .txt).wav"
done

# Process all PDFs  
for file in documents/*.pdf; do
    dts mp3 -s "$file" -o "audio/$(basename "$file" .pdf).mp3"
done
```

### Python API Examples

#### Convert Multiple Documents

```python
import os
from dts.speaker import DocumentToSpeech

tts = DocumentToSpeech()

# Convert all PDFs in a directory
pdf_dir = "documents/"
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        input_path = os.path.join(pdf_dir, filename)
        output_path = f"audio/{filename[:-4]}.wav"
        tts.doc_to_wav(input_path, output_path)
        print(f"Converted {filename} to audio")
```

#### Batch Processing with Progress

```python
documents = ["doc1.pdf", "doc2.txt", "doc3.html"]
tts = DocumentToSpeech()

for i, doc in enumerate(documents, 1):
    print(f"Processing document {i}/{len(documents)}: {doc}")
    output_file = f"audio/output_{i}.mp3"
    tts.doc_to_mp3(doc, output_file, progress=True)
```

#### Custom Voice with Text Input

```python
from dts.vibevoice_tts import TextToSpeech

# Direct text-to-speech with custom voice
tts = TextToSpeech(sample_voice_file="my_voice.wav")
text = "This is a test of custom voice synthesis."
tts.convert_text_to_wav(text, "custom_output.wav")
```

### Combining CLI and Python API

```python
import subprocess
import os

# Use CLI for bulk conversion, then Python API for custom processing
def hybrid_processing():
    # Step 1: Use CLI for fast batch conversion
    subprocess.run([
        "dts", "wav", 
        "-s", "documents/", 
        "-o", "temp_audio/"
    ])
    
    # Step 2: Use Python API for custom post-processing
    from dts.vibevoice_tts import TextToSpeech
    tts = TextToSpeech()
    
    # Custom voice processing on generated files
    for wav_file in os.listdir("temp_audio/"):
        if wav_file.endswith(".wav"):
            # Additional processing or format conversion
            mp3_file = wav_file.replace(".wav", ".mp3")
            subprocess.run([
                "dts", "wav2mp3", 
                "-s", f"temp_audio/{wav_file}", 
                "-o", f"final_audio/{mp3_file}"
            ])
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

[Add your license information here]

## Dependencies

Key dependencies include:

- **PyTorch**: Deep learning framework
- **VibeVoice**: TTS model for speech synthesis  
- **PyMuPDF**: PDF text extraction
- **BeautifulSoup4**: HTML parsing
- **Docling**: Universal document processing
- **librosa**: Audio processing
- **pydub**: Audio format conversion
- **Click**: Command-line interface framework

See `pyproject.toml` for the complete dependency list.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_chars` parameter
2. **Unsupported Document Format**: Check if format is in supported list
3. **Audio Quality Issues**: Verify voice sample quality and format
4. **Slow Processing**: Enable GPU acceleration if available
5. **CLI Command Not Found**: Ensure package is installed in the correct environment
6. **Permission Errors**: Check file permissions for input and output directories

### Getting Help

- Use `dts --help` for CLI usage information
- Use `dts <command> --help` for specific command help
- Check the documentation for parameter details
- Verify input document format compatibility
- Ensure proper PyTorch installation for your system
- Monitor memory usage during processing

---

*This library leverages the VibeVoice model for high-quality text-to-speech synthesis. For more information about the underlying TTS technology, visit the VibeVoice project.*
