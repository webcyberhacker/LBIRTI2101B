# Wazo 

This project develops a CNN algorithm to identify birds from records.
among the 10 most present species in Belgium (source: Natagora):

- Merle noir (Turdus merula)
- Mésange charbonnière (Parus major)
- Mésange bleue (Cyanistes caeruleus)
- Rouge-gorge familier (Erithacus rubecula)
- Moineau domestique (Passer domesticus)
- Pigeon ramier (Columba palumbus)
- Étourneau sansonnet (Sturnus vulgaris)
- Pinson des arbres (Fringilla coelebs)
- Tourterelle turque
- Pie bavarde

## Tools

- Xeno-Canto API to extract recordings from birds
- Pytorch to build the CNN model

## Tech stack

- UV for library management
- PyTorch for CNN model
- librosa for audio processing
- scikit-learn for data splitting

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Preprocess recordings (converts audio to mel-spectrograms):
```bash
python -m src.preprocess_recordings
```

This will:
- Load all audio files from `recordings/` directory
- Convert them to mel-spectrograms (128x130 by default)
- Save preprocessed data to `data/preprocessed/`

3. Create train/validation/test splits:
```bash
python -m src.create_splits
```

This creates splits in `data/dataset/` with subdirectories:
- `train/` (70% by default)
- `val/` (15% by default)
- `test/` (15% by default)

## Usage

### Preprocessing

The preprocessing pipeline:
- Converts audio files (mp3, wav) to mel-spectrograms
- Normalizes audio length to 3 seconds (trims or pads)
- Resizes spectrograms to fixed dimensions (128x130)
- Saves as numpy arrays (.npy files)

### Loading Data

Use the `BirdDataset` class to load preprocessed data:

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import BirdDataset

# Load dataset
dataset = BirdDataset(Path("data/preprocessed"))

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Use in training loop
for spectrograms, labels in dataloader:
    # spectrograms shape: (batch_size, 1, 128, 130)
    # labels shape: (batch_size,)
    pass
```

See `example_usage.py` for a complete example.

### Training

Train the CNN model:

```bash
python -m src.train
```

Options:
- `--data-dir`: Directory with train/val/test splits (default: `data/dataset`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--output-dir`: Directory to save models (default: `models`)

The training script will:
- Train the model on the training set
- Validate on the validation set after each epoch
- Save the best model (highest validation accuracy) to `models/best_model.pth`
- Save checkpoints every 10 epochs
- Save training history to `models/training_history.json`

### Evaluation

Evaluate the trained model on the test set:

```bash
python -m src.evaluate
```

Options:
- `--model-path`: Path to model checkpoint (default: `models/best_model.pth`)
- `--test-dir`: Directory containing test data (default: `data/dataset/test`)

This will print:
- Test accuracy and loss
- Classification report (precision, recall, F1-score per class)
- Confusion matrix

## Model Architecture

The CNN model consists of:
- 4 convolutional blocks with batch normalization and dropout
- Each block has 2 convolutional layers followed by max pooling
- Feature maps: 32 → 64 → 128 → 256
- 3 fully connected layers (512 → 256 → num_classes)
- Total parameters: ~2.5M

## Questions

Filter choices:

- Quantity of recordings?
- Quality? -> Do I want noise or not in my training set? And in my validation set?
- Length?
- Location?
- Number? Start with a 100 for fast iteration, then train another one on all recording available, and compare results.
