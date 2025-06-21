# Speech-to-Text Recognition System

A comprehensive speech-to-text system supporting multiple recognition engines including Google Speech Recognition, CMU Sphinx, and Wav2Vec2.

## Features

- **Multiple Recognition Engines**: Google Speech Recognition (online), CMU Sphinx (offline), Wav2Vec2 (offline)
- **Real-time Transcription**: Live speech-to-text conversion
- **File Transcription**: Process audio files in various formats
- **Batch Processing**: Transcribe multiple files at once
- **Ambient Noise Adjustment**: Automatic microphone calibration
- **Multiple Audio Formats**: Support for WAV, FLAC, MP3, and more

## Installation

### 1. Install Python Dependencies

```bash
# Core dependencies
pip install SpeechRecognition pyaudio

# For file format support
pip install pydub

# For Sphinx (offline recognition)
pip install pocketsphinx

# For Wav2Vec2 (advanced offline recognition)
pip install torch torchaudio transformers

# For audio processing
pip install librosa soundfile
```

### 2. System Dependencies

#### Windows
```bash
# PyAudio might need Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# For pocketsphinx, you might need:
pip install pipwin
pipwin install pyaudio
```

#### macOS
```bash
# Install portaudio for PyAudio
brew install portaudio
pip install pyaudio

# If you get compilation errors:
export CPPFLAGS=-I/opt/homebrew/include
export LDFLAGS=-L/opt/homebrew/lib
pip install pyaudio
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev
sudo apt-get install flac  # For FLAC support
sudo apt-get install ffmpeg  # For various audio formats

# Install Python packages
pip install SpeechRecognition pyaudio pocketsphinx
```

### 3. Complete Requirements File

Create a `requirements.txt` file:

```
SpeechRecognition>=3.10.0
pyaudio>=0.2.11
pydub>=0.25.1
pocketsphinx>=0.1.15
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.21.0
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage

```python
from speech_to_text_system import SpeechToTextSystem

# Initialize the system
stt = SpeechToTextSystem()

# Record and transcribe
stt.adjust_for_ambient_noise()
audio = stt.record_audio(phrase_time_limit=5)
text = stt.transcribe_with_google(audio)
print(f"Transcribed: {text}")
```

### Transcribe Audio File

```python
# Transcribe a single file
text = stt.transcribe_audio_file("audio.wav", method="google")
print(text)

# Batch transcribe multiple files
files = ["audio1.wav", "audio2.mp3", "audio3.flac"]
results = stt.batch_transcribe(files, method="wav2vec")
```

### Real-time Transcription

```python
# Start real-time transcription for 60 seconds
stt.real_time_transcription(method="google", duration=60)
```

## Supported Audio Formats

- **WAV** (recommended)
- **FLAC** 
- **MP3** (requires pydub)
- **M4A** (requires pydub)
- **OGG** (requires pydub)

## Recognition Methods Comparison

| Method | Online/Offline | Accuracy | Speed | Notes |
|--------|---------------|----------|-------|-------|
| Google | Online | High | Fast | Requires internet |
| Sphinx | Offline | Medium | Fast | Works offline |
| Wav2Vec2 | Offline | High | Medium | Large model, high accuracy |

## Troubleshooting

### Common Issues

1. **PyAudio Installation Issues**
   ```bash
   # Windows
   pip install pipwin
   pipwin install pyaudio
   
   # macOS with homebrew
   brew install portaudio
   
   # Linux
   sudo apt-get install python3-pyaudio
   ```

2. **Microphone Not Working**
   ```python
   # List available microphones
   import speech_recognition as sr
   for index, name in enumerate(sr.Microphone.list_microphone_names()):
       print(f"Microphone {index}: {name}")
   
   # Use specific microphone
   mic = sr.Microphone(device_index=1)  # Use microphone at index 1
   ```

3. **Wav2Vec2 Model Download Issues**
   ```python
   # Pre-download the model
   from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
   
   model_name = "facebook/wav2vec2-base-960h"
   tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
   model = Wav2Vec2ForCTC.from_pretrained(model_name)
   ```

4. **Audio Format Issues**
   ```python
   # Convert audio format using pydub
   from pydub import AudioSegment
   
   audio = AudioSegment.from_mp3("input.mp3")
   audio.export("output.wav", format="wav")
   ```

## Performance Tips

1. **For Best Accuracy**:
   - Use quiet environment
   - Speak clearly and at moderate pace
   - Use good quality microphone
   - Adjust for ambient noise before recording

2. **For Real-time Processing**:
   - Use Google Speech Recognition for best results
   - Keep phrase_time_limit reasonable (3-5 seconds)
   - Handle exceptions gracefully

3. **For Offline Use**:
   - Use Wav2Vec2 for best offline accuracy
   - Sphinx is faster but less accurate
   - Pre-download Wav2Vec2 models

## API Integration

The system can be easily integrated into other applications:

```python
class MyApp:
    def __init__(self):
        self.stt = SpeechToTextSystem()
    
    def process_audio(self, audio_file):
        try:
            return self.stt.transcribe_audio_file(audio_file)
        except Exception as e:
            return f"Error: {e}"
```

## License and Credits

This system uses several open-source libraries:
- SpeechRecognition library
- PyAudio for microphone access
- Facebook's Wav2Vec2 models
- CMU Sphinx for offline recognition

Please check individual library licenses for commercial use.
