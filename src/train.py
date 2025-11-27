"""Training script for bird audio classification CNN."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

from src.model import create_model
from src.dataset import BirdDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for spectrograms, labels in tqdm(dataloader, desc="Training"):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(dataloader, desc="Validating"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def plot_training_results(history, save_dir):
    """Génère et sauvegarde les courbes d'apprentissage."""
    epochs = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(15, 6))

    # Graphique 1 : Accuracy (Précision)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graphique 2 : Loss (Perte/Erreur)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarde
    plt.tight_layout()
    plot_path = save_dir / "training_plots.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✓ Plots saved to: {plot_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train bird audio classification CNN")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/dataset",
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Train or validation directories not found in {data_dir}")
        return
    
    print("Loading datasets...")
    train_dataset = BirdDataset(train_dir)
    val_dataset = BirdDataset(val_dir, bird_classes=train_dataset.bird_classes)
    
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.get_class_names()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 on Mac/Windows, or higher on Linux
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=num_classes, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    current_lr = args.learning_rate
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'class_names': train_dataset.get_class_names(),
            }
            
            best_model_path = output_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'class_names': train_dataset.get_class_names(),
            }
            torch.save(checkpoint, checkpoint_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\nGenerating training plots...")
    plot_training_results(history, output_dir)
    # -------------------

    print("\n" + "=" * 60)
    print("Training completed!")
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Model saved to: {output_dir}")
    print(f"Best model: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

