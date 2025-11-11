"""Audio preprocessing for CNN model training."""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocess audio files into mel-spectrograms for CNN training."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 3.0,
        target_shape: Tuple[int, int] = (128, 130),
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel filter banks
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            duration: Target duration in seconds (audio will be trimmed/padded)
            target_shape: Target shape for spectrogram (height, width)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_shape = target_shape
        
    def load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal as numpy array
        """
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if loading fails
            return np.zeros(int(self.sample_rate * self.duration))
    
    def trim_or_pad(self, audio: np.ndarray) -> np.ndarray:
        """
        Trim or pad audio to target duration.
        
        Args:
            audio: Audio signal
            
        Returns:
            Audio trimmed/padded to target duration
        """
        target_length = int(self.sample_rate * self.duration)
        
        if len(audio) > target_length:
            # Trim to center
            start = (len(audio) - target_length) // 2
            return audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        else:
            return audio
    
    def audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel-spectrogram as numpy array
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_db
    
    def resize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Resize spectrogram to target shape.
        
        Args:
            spec: Spectrogram array
            
        Returns:
            Resized spectrogram
        """
        from scipy.ndimage import zoom
        
        height, width = spec.shape
        target_height, target_width = self.target_shape
        
        # Calculate zoom factors
        zoom_y = target_height / height
        zoom_x = target_width / width
        
        # Resize using zoom
        resized = zoom(spec, (zoom_y, zoom_x), order=1)
        
        return resized
    
    def process_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Process a single audio file into a mel-spectrogram.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed mel-spectrogram or None if processing fails
        """
        try:
            # Load audio
            audio = self.load_audio(file_path)
            
            # Trim or pad to target duration
            audio = self.trim_or_pad(audio)
            
            # Convert to mel-spectrogram
            mel_spec = self.audio_to_melspectrogram(audio)
            
            # Resize to target shape
            mel_spec = self.resize_spectrogram(mel_spec)
            
            return mel_spec
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def process_directory(
        self,
        recordings_dir: Path,
        output_dir: Path,
        bird_name: str,
    ) -> list[Path]:
        """
        Process all audio files in a directory.
        
        Args:
            recordings_dir: Directory containing audio files
            output_dir: Directory to save preprocessed files
            bird_name: Name of the bird species
            
        Returns:
            List of paths to saved preprocessed files
        """
        output_dir = Path(output_dir) / bird_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(recordings_dir.glob("*.mp3")) + list(recordings_dir.glob("*.wav")) + list(recordings_dir.glob("*.MP3"))
        
        saved_paths = []
        
        for audio_file in audio_files:
            print(f"Processing {audio_file.name}...")
            mel_spec = self.process_file(audio_file)
            
            if mel_spec is not None:
                # Save as numpy array
                output_path = output_dir / f"{audio_file.stem}.npy"
                np.save(output_path, mel_spec)
                saved_paths.append(output_path)
        
        return saved_paths

