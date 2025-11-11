"""PyTorch Dataset for bird audio classification."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import LabelEncoder


class BirdDataset(Dataset):
    """Dataset for bird audio classification."""
    
    def __init__(
        self,
        data_dir: Path,
        bird_classes: Optional[list[str]] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize bird dataset.
        
        Args:
            data_dir: Directory containing preprocessed data (with subdirectories for each bird)
            bird_classes: List of bird class names. If None, will be inferred from subdirectories.
            transform: Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all preprocessed files
        self.samples = []
        self.labels = []
        
        # Get bird classes from subdirectories if not provided
        if bird_classes is None:
            bird_classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        self.bird_classes = bird_classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.bird_classes)
        
        # Load all samples
        for bird_class in self.bird_classes:
            bird_dir = self.data_dir / bird_class
            if not bird_dir.exists():
                continue
                
            label = self.label_encoder.transform([bird_class])[0]
            
            # Find all .npy files
            npy_files = list(bird_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                self.samples.append(npy_file)
                self.labels.append(label)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.bird_classes)} classes")
        print(f"Classes: {self.bird_classes}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (spectrogram, label)
        """
        # Load preprocessed spectrogram
        spectrogram = np.load(self.samples[idx])
        
        # Convert to tensor and add channel dimension
        spectrogram = torch.from_numpy(spectrogram).float()
        spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension: (1, H, W)
        
        label = self.labels[idx]
        
        # Apply transform if provided
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label
    
    def get_class_names(self) -> list[str]:
        """Get list of class names."""
        return self.bird_classes
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.bird_classes)

