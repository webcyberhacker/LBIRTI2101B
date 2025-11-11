"""Create train/validation/test splits from preprocessed data."""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


def create_splits(
    preprocessed_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
):
    """
    Create train/validation/test splits from preprocessed data.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data (with subdirectories for each bird)
        output_dir: Directory to save split data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed for reproducibility
    """
    preprocessed_dir = Path(preprocessed_dir)
    output_dir = Path(output_dir)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each bird species
    bird_dirs = sorted([d for d in preprocessed_dir.iterdir() if d.is_dir()])
    
    if not bird_dirs:
        print(f"No bird directories found in {preprocessed_dir}")
        return
    
    print(f"Found {len(bird_dirs)} bird species")
    print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print("-" * 50)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for bird_dir in bird_dirs:
        bird_name = bird_dir.name
        
        # Get all .npy files for this bird
        npy_files = sorted(list(bird_dir.glob("*.npy")))
        
        if len(npy_files) == 0:
            print(f"  {bird_name}: No files found, skipping")
            continue
        
        # Create bird subdirectories in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / bird_name).mkdir(parents=True, exist_ok=True)
        
        # First split: train vs (val + test)
        train_files, temp_files = train_test_split(
            npy_files,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_size),
            random_state=random_state,
        )
        
        # Copy files to respective directories
        for file_path in train_files:
            shutil.copy2(file_path, train_dir / bird_name / file_path.name)
            total_train += 1
        
        for file_path in val_files:
            shutil.copy2(file_path, val_dir / bird_name / file_path.name)
            total_val += 1
        
        for file_path in test_files:
            shutil.copy2(file_path, test_dir / bird_name / file_path.name)
            total_test += 1
        
        print(f"  {bird_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    print("\n" + "=" * 50)
    print(f"Total files:")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    print(f"  Test: {total_test}")
    print(f"\nSplits saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits from preprocessed data")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="data/preprocessed",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset",
        help="Directory to save split data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of test data",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    create_splits(
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

