"""Example usage of the preprocessing and dataset modules."""

from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import BirdDataset
from src.preprocessing import AudioPreprocessor


def example_preprocessing():
    """Example: Preprocess recordings."""
    print("=" * 50)
    print("Example: Preprocessing Recordings")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_mels=128,
        duration=3.0,
    )
    
    # Process a single bird species
    recordings_dir = Path("recordings/Parus_major")
    output_dir = Path("data/preprocessed")
    
    if recordings_dir.exists():
        print(f"\nProcessing {recordings_dir}...")
        saved_paths = preprocessor.process_directory(
            recordings_dir=recordings_dir,
            output_dir=output_dir,
            bird_name="Parus_major",
        )
        print(f"Processed {len(saved_paths)} files")
    else:
        print(f"Directory {recordings_dir} not found")


def example_dataset():
    """Example: Loading dataset with PyTorch DataLoader."""
    print("\n" + "=" * 50)
    print("Example: Loading Dataset")
    print("=" * 50)
    
    preprocessed_dir = Path("data/preprocessed")
    
    if not preprocessed_dir.exists():
        print(f"Preprocessed directory {preprocessed_dir} not found.")
        print("Run preprocessing first!")
        return
    
    # Create dataset
    dataset = BirdDataset(preprocessed_dir)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Set to 0 on Windows/Mac, or higher on Linux
    )
    
    # Get a batch
    print("\nLoading a batch...")
    for batch_idx, (spectrograms, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Spectrogram shape: {spectrograms.shape}")  # (batch_size, 1, height, width)
        print(f"  Labels shape: {labels.shape}")  # (batch_size,)
        print(f"  Labels: {labels.tolist()}")
        
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    print("\nDataset is ready for CNN training!")


if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Step 1: Preprocess recordings (run this first)
    # example_preprocessing()
    
    # Step 2: Load dataset (run this after preprocessing)
    example_dataset()

