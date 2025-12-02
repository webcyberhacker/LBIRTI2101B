import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
from scipy.ndimage import zoom  # Required for resizing in AudioPreprocessor

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. PREPROCESSING
# ==============================================================================

class AudioPreprocessor:
    """Preprocess audio files into mel-spectrograms for CNN inference."""

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
        """Load audio file and resample to target sample rate."""
        try:
            # Load audio file, forcing mono
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(int(self.sample_rate * self.duration))

    def trim_or_pad(self, audio: np.ndarray) -> np.ndarray:
        """Trim or pad audio to target duration."""
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
        """Convert audio to mel-spectrogram."""
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
        """Resize spectrogram to target shape."""
        height, width = spec.shape
        target_height, target_width = self.target_shape

        zoom_y = target_height / height
        zoom_x = target_width / width

        resized = zoom(spec, (zoom_y, zoom_x), order=1)
        return resized

    def process_file(self, file_path: Path) -> Optional[np.ndarray]:
        """Process a single audio file into a mel-spectrogram."""
        try:
            audio = self.load_audio(file_path)
            audio = self.trim_or_pad(audio)
            mel_spec = self.audio_to_melspectrogram(audio)
            mel_spec = self.resize_spectrogram(mel_spec)
            return mel_spec
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


# ==============================================================================
# 2. MODEL ARCHITECTURE 
# ==============================================================================

class BirdCNN(nn.Module):
    """CNN model for classifying bird species from mel-spectrograms."""

    def __init__(self, num_classes: int, input_channels: int = 1):
        super(BirdCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.4)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.4)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.4)

        # Block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.4)

        # Fully Connected
        self.fc_input_size = 256 * 8 * 8
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)  # Note: Dropout is active only in train mode

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)

        return x


# ==============================================================================
# 3. INFERENCE ENGINE (The glue that puts it all together)
# ==============================================================================

class BirdClassifier:
    """Wrapper class to handle model loading and prediction."""

    def __init__(self, model_path: str, labels: Optional[List[str]] = None):
        """
        Args:
            model_path: Path to the .pth file
            labels: Optional list. If None, we try to load them from the file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor()

        # 1. On charge le fichier BRUT d'abord
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"Fichier chargé. Clés trouvées : {checkpoint.keys()}")
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le fichier .pth : {e}")

        # 2. Gestion intelligente : Est-ce un Checkpoint ou des poids simples ?
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # CAS 1 : C'est un Checkpoint (votre cas actuel)
            print("Détection d'un Checkpoint complet.")
            state_dict = checkpoint['model_state_dict']

            # TENTATIVE DE RÉCUPÉRATION DES NOMS D'OISEAUX AUTOMATIQUE
            if 'class_names' in checkpoint:
                self.labels = checkpoint['class_names']
                print(f"Espèces trouvées dans le fichier : {self.labels}")
            elif labels is not None:
                self.labels = labels
            else:
                raise ValueError("Aucune liste d'espèces trouvée dans le fichier ni fournie en argument.")

        else:
            # CAS 2 : C'est juste des poids (cas classique)
            state_dict = checkpoint
            if labels is None:
                raise ValueError("Pour ce type de fichier, vous devez fournir la liste 'labels'.")
            self.labels = labels

        # 3. Initialisation du modèle avec le bon nombre de classes
        self.model = BirdCNN(num_classes=len(self.labels))

        # 4. Chargement des poids
        try:
            self.model.load_state_dict(state_dict)
            print("Poids du modèle chargés avec succès !")
        except Exception as e:
            print(f"Erreur de chargement des poids : {e}")
            raise e

        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_file_path: str) -> Dict[str, any]:
        """
        Predict the bird species from an audio file.

        Returns:
            Dictionary containing the predicted label and confidence score.
        """
        # 1. Preprocess the audio
        spec = self.preprocessor.process_file(Path(audio_file_path))

        if spec is None:
            return {"error": "Failed to process audio file"}

        # 2. Convert to Tensor and add Batch/Channel dimensions
        # Shape becomes: (1, 1, 128, 130) -> (Batch, Channel, Height, Width)
        tensor = torch.from_numpy(spec).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        # 3. Inference
        with torch.no_grad():  # Disable gradient calculation for speed
            outputs = self.model(tensor)

            # Apply Softmax to get probabilities (percentages)
            probabilities = F.softmax(outputs, dim=1)

            # Get the winner
            score, predicted_idx = torch.max(probabilities, 1)

            idx = predicted_idx.item()
            confidence = score.item()

        return {
            "label": self.labels[idx],
            "confidence": confidence,
            "all_probabilities": {self.labels[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }