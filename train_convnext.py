import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import time
from pathlib import Path
import argparse
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def create_data_loaders(data_dir, batch_size=16, seed=42):
    """Create train, validation, and test data loaders (70/15/15 split)"""
    
    # ConvNeXt standard input is 224x224
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Load full dataset to get indices for split
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    total_size = len(full_dataset)
    
    # Calculate split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Generate indices
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create datasets with appropriate transforms
    # We re-instantiate ImageFolder with specific transforms for each split
    train_data = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_subset.indices)
    val_data = Subset(datasets.ImageFolder(data_dir, transform=val_test_transform), val_subset.indices)
    test_data = Subset(datasets.ImageFolder(data_dir, transform=val_test_transform), test_subset.indices)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    return train_loader, val_loader, test_loader, class_names

def create_convnext_model(model_name, num_classes, device):
    """Create and configure ConvNeXt model"""
    print(f"Initializing {model_name}...")
    
    if model_name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights=weights)
    elif model_name == 'convnext_large':
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1

def evaluate_test_set(model, dataloader, class_names, device, save_dir):
    print("\n" + "="*70)
    print("🧪 Running Final Evaluation on Test Set...")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\nClassification Report:")
    print(report_text)
    
    # Save Report
    with open(save_dir / 'test_classification_report.txt', 'w') as f:
        f.write(report_text)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / 'test_confusion_matrix.png')
    print(f"✓ Confusion matrix saved to {save_dir / 'test_confusion_matrix.png'}")
    
    return report

def train_model():
    parser = argparse.ArgumentParser(description='Train ConvNeXt on Flower Dataset')
    parser.add_argument('--model_name', type=str, default='convnext_base', 
                        choices=['convnext_base', 'convnext_large'])
    parser.add_argument('--data_dir', type=str, default=r'd:/archive/flowers',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print(f"Flower Classification Training - {args.model_name}")
    print("="*70)
    print(f"Device: {device}")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Load Data
    print("\nLoading dataset (70/15/15 split)...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes: {class_names}")
    
    # Model
    print(f"\nCreating {args.model_name} model...")
    model = create_convnext_model(args.model_name, len(class_names), device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    best_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        val_loss, val_acc, val_p, val_r, val_f1 = validate_epoch(model, val_loader, criterion, device)
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_names': class_names,
                'model_name': args.model_name
            }, save_path / f'{args.model_name}_best.pth')
            print(f"Saved best model (Val Acc: {val_acc:.4f})")
            
    print("\n" + "="*70)
    print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
    
    # Load best model for testing
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(save_path / f'{args.model_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluate_test_set(model, test_loader, class_names, device, save_path)
    
    # Save Class Names
    with open(save_path / f'{args.model_name}_class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

if __name__ == "__main__":
    train_model()
