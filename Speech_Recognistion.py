import speech_recognition as sr
import pyaudio
import wave
import tempfile
import os
from pathlib import Path
import threading
import time
from typing import Optional, List, Dict
import logging

# For Wav2Vec2 (optional - requires transformers and torch)
try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False
    print("Wav2Vec2 dependencies not available. Install with: pip install torch torchaudio transformers")

class SpeechToTextSystem:
    """
    A comprehensive speech-to-text system supporting multiple recognition engines.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.setup_logging()
        
        # Initialize Wav2Vec2 if available
        self.wav2vec_model = None
        self.wav2vec_tokenizer = None
        if WAV2VEC_AVAILABLE:
            self.load_wav2vec_model()
    
    def setup_logging(self):
        """Setup logging for the system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_wav2vec_model(self):
        """Load pre-trained Wav2Vec2 model"""
        try:
            model_name = "facebook/wav2vec2-base-960h"
            self.wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.logger.info("Wav2Vec2 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Wav2Vec2 model: {e}")
            self.wav2vec_model = None
            self.wav2vec_tokenizer = None
    
    def adjust_for_ambient_noise(self, duration: float = 1.0):
        """Adjust recognizer sensitivity to ambient noise"""
        try:
            with self.microphone as source:
                self.logger.info(f"Adjusting for ambient noise... Please wait {duration} seconds")
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self.logger.info("Ambient noise adjustment complete")
        except Exception as e:
            self.logger.error(f"Error adjusting for ambient noise: {e}")
    
    def record_audio(self, duration: Optional[float] = None, phrase_time_limit: Optional[float] = None) -> sr.AudioData:
        """
        Record audio from microphone
        
        Args:
            duration: Maximum recording duration in seconds
            phrase_time_limit: Maximum time to wait for a phrase
        
        Returns:
            AudioData object containing the recorded audio
        """
        try:
            with self.microphone as source:
                self.logger.info("Recording... Speak now!")
                if duration:
                    audio = self.recognizer.record(source, duration=duration)
                else:
                    audio = self.recognizer.listen(
                        source, 
                        phrase_time_limit=phrase_time_limit,
                        timeout=10
                    )
            self.logger.info("Recording complete")
            return audio
        except sr.WaitTimeoutError:
            self.logger.error("Recording timeout - no speech detected")
            raise
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            raise
    
    def transcribe_with_google(self, audio_data: sr.AudioData, language: str = "en-US") -> str:
        """
        Transcribe audio using Google Speech Recognition (requires internet)
        
        Args:
            audio_data: AudioData object to transcribe
            language: Language code for recognition
        
        Returns:
            Transcribed text
        """
        try:
            text = self.recognizer.recognize_google(audio_data, language=language)
            self.logger.info("Google Speech Recognition successful")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Google Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Google Speech Recognition error: {e}")
            raise
    
    def transcribe_with_sphinx(self, audio_data: sr.AudioData) -> str:
        """
        Transcribe audio using CMU Sphinx (offline)
        
        Args:
            audio_data: AudioData object to transcribe
        
        Returns:
            Transcribed text
        """
        try:
            text = self.recognizer.recognize_sphinx(audio_data)
            self.logger.info("Sphinx Speech Recognition successful")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Sphinx could not understand audio")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Sphinx error: {e}")
            raise
    
    def transcribe_with_wav2vec(self, audio_file_path: str) -> str:
        """
        Transcribe audio using Wav2Vec2 model (offline)
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        if not WAV2VEC_AVAILABLE or not self.wav2vec_model:
            raise ValueError("Wav2Vec2 model not available")
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_file_path)
            
            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Get model predictions
            with torch.no_grad():
                logits = self.wav2vec_model(waveform).logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec_tokenizer.decode(predicted_ids[0])
            
            self.logger.info("Wav2Vec2 transcription successful")
            return transcription.lower()
        
        except Exception as e:
            self.logger.error(f"Wav2Vec2 transcription error: {e}")
            raise
    
    def transcribe_audio_file(self, file_path: str, method: str = "google") -> str:
        """
        Transcribe audio from file
        
        Args:
            file_path: Path to audio file
            method: Recognition method ("google", "sphinx", "wav2vec")
        
        Returns:
            Transcribed text
        """
        try:
            # Load audio file
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
            
            if method == "google":
                return self.transcribe_with_google(audio_data)
            elif method == "sphinx":
                return self.transcribe_with_sphinx(audio_data)
            elif method == "wav2vec":
                return self.transcribe_with_wav2vec(file_path)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            self.logger.error(f"Error transcribing file {file_path}: {e}")
            raise
    
    def real_time_transcription(self, method: str = "google", duration: int = 60):
        """
        Perform real-time speech transcription
        
        Args:
            method: Recognition method to use
            duration: Total duration to run (seconds)
        """
        self.logger.info(f"Starting real-time transcription for {duration} seconds...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                try:
                    # Record short audio segments
                    audio = self.record_audio(phrase_time_limit=3)
                    
                    # Transcribe based on method
                    if method == "google":
                        text = self.transcribe_with_google(audio)
                    elif method == "sphinx":
                        text = self.transcribe_with_sphinx(audio)
                    else:
                        # For wav2vec, we need to save to file first
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            with wave.open(tmp_file.name, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(audio.sample_width)
                                wav_file.setframerate(audio.sample_rate)
                                wav_file.writeframes(audio.get_raw_data())
                            text = self.transcribe_with_wav2vec(tmp_file.name)
                            os.unlink(tmp_file.name)
                    
                    if text.strip():
                        print(f"[{time.strftime('%H:%M:%S')}] Transcribed: {text}")
                
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Real-time transcription error: {e}")
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Real-time transcription stopped by user")
    
    def batch_transcribe(self, audio_files: List[str], method: str = "google") -> Dict[str, str]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_files: List of audio file paths
            method: Recognition method to use
        
        Returns:
            Dictionary mapping file paths to transcriptions
        """
        results = {}
        
        for file_path in audio_files:
            try:
                self.logger.info(f"Transcribing {file_path}...")
                text = self.transcribe_audio_file(file_path, method)
                results[file_path] = text
                print(f"✓ {file_path}: {text}")
            except Exception as e:
                self.logger.error(f"Failed to transcribe {file_path}: {e}")
                results[file_path] = f"ERROR: {str(e)}"
        
        return results
    
    def save_transcription(self, text: str, output_file: str):
        """Save transcription to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Transcription saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving transcription: {e}")
            raise

def main():
    """Main function demonstrating the speech-to-text system"""
    
    # Initialize the system
    stt = SpeechToTextSystem()
    
    print("Speech-to-Text Recognition System")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Record and transcribe (Google)")
        print("2. Record and transcribe (Sphinx - offline)")
        print("3. Transcribe audio file")
        print("4. Real-time transcription")
        print("5. Batch transcribe files")
        if WAV2VEC_AVAILABLE:
            print("6. Transcribe with Wav2Vec2")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        try:
            if choice == "1":
                print("\nPreparing to record...")
                stt.adjust_for_ambient_noise()
                audio = stt.record_audio(phrase_time_limit=5)
                text = stt.transcribe_with_google(audio)
                print(f"\nTranscription: {text}")
                
                save = input("Save transcription? (y/n): ").lower()
                if save == 'y':
                    filename = input("Enter filename (without extension): ") + ".txt"
                    stt.save_transcription(text, filename)
            
            elif choice == "2":
                print("\nPreparing to record...")
                stt.adjust_for_ambient_noise()
                audio = stt.record_audio(phrase_time_limit=5)
                text = stt.transcribe_with_sphinx(audio)
                print(f"\nTranscription: {text}")
            
            elif choice == "3":
                file_path = input("Enter audio file path: ").strip()
                if not os.path.exists(file_path):
                    print("File not found!")
                    continue
                
                method = input("Choose method (google/sphinx/wav2vec): ").lower()
                if method not in ["google", "sphinx", "wav2vec"]:
                    method = "google"
                
                text = stt.transcribe_audio_file(file_path, method)
                print(f"\nTranscription: {text}")
            
            elif choice == "4":
                method = input("Choose method (google/sphinx): ").lower()
                if method not in ["google", "sphinx"]:
                    method = "google"
                
                duration = int(input("Duration in seconds (default 30): ") or "30")
                print(f"\nStarting real-time transcription with {method}...")
                print("Press Ctrl+C to stop")
                stt.real_time_transcription(method, duration)
            
            elif choice == "5":
                files_input = input("Enter audio file paths (comma-separated): ")
                files = [f.strip() for f in files_input.split(",")]
                method = input("Choose method (google/sphinx/wav2vec): ").lower()
                if method not in ["google", "sphinx", "wav2vec"]:
                    method = "google"
                
                results = stt.batch_transcribe(files, method)
                print("\nBatch transcription results:")
                for file, text in results.items():
                    print(f"{file}: {text}")
            
            elif choice == "6" and WAV2VEC_AVAILABLE:
                file_path = input("Enter audio file path: ").strip()
                if not os.path.exists(file_path):
                    print("File not found!")
                    continue
                
                text = stt.transcribe_with_wav2vec(file_path)
                print(f"\nWav2Vec2 Transcription: {text}")
            
            elif choice == "0":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice!")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()