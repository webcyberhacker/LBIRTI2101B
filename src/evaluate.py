"""Evaluate trained model on test set."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.model import create_model
from src.dataset import BirdDataset

import os
import numpy as np


def evaluate(model_path: Path, test_dir: Path, device: str = "auto", batch_size: int = 16):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_path: Path to saved model checkpoint
        test_dir: Directory containing test data
        device: Device to use
        batch_size: Batch size for evaluation
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    print(f"Model trained for {num_classes} classes: {class_names}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Create model and load weights
    model = create_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset
    print(f"\nLoading test dataset from {test_dir}...")
    test_dataset = BirdDataset(test_dir, bird_classes=class_names)
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(test_loader, desc="Evaluating"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Correct: {correct}/{total}")
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
    ))
    
    # Confusion matrix
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds)
    print("\nPredicted ->")
    print("Actual ↓", end="")
    for name in class_names:
        print(f"\t{name[:10]:<10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:10]:<10}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm[i][j]:<10}", end="")
        print()

    
    # Save results to /results/
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save confusion matrix
    cm_path = results_dir / "confusion_matrix.csv"
    np.savetxt(cm_path, cm, fmt='%d', delimiter=',')

    print(f"Confusion matrix saved to: {cm_path}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained bird audio classification CNN")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/test_3/00001_32_80/best_model.pth",
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/dataset/test",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        return
    
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} not found")
        return
    
    evaluate(
        model_path=model_path,
        test_dir=test_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

