import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def load_convnext_model(model_path, model_name, num_classes, device):
    """Load trained ConvNeXt model"""
    print(f"Loading {model_name} from {model_path}...")
    
    if model_name == 'convnext_base':
        weights = None
        model = models.convnext_base(weights=weights) 
    elif model_name == 'convnext_large':
        weights = None
        model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, class_names, device, save_dir):
    all_preds = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 70)
    
    # Detailed Report
    print("\nDetailed Classification Report:")
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_text)
    
    # Save Report
    with open(os.path.join(save_dir, 'external_test_report.txt'), 'w') as f:
        f.write(report_text)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'external_test_confusion_matrix.png')
    plt.savefig(save_path)
    print(f"\nConfusion matrix saved to {save_path}")
    
    return all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description='Test ConvNeXt Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data_dir', type=str, default=r'd:/archive/flowers', help='Path to data folder to test on')
    parser.add_argument('--model_name', type=str, default='convnext_base', choices=['convnext_base', 'convnext_large'])
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
        
    # Load Checkpoint to get class names
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            print(f"Loaded {len(class_names)} classes from checkpoint.")
        else:
            print("⚠ Class names not in checkpoint, inferring from data directory...")
            # This relies on the test dir having the SAME structure/class order
            temp_dataset = datasets.ImageFolder(args.data_dir)
            class_names = temp_dataset.classes
    except Exception as e:
        print(f"❌ Error loading checkpoint metadata: {e}")
        sys.exit(1)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    try:
        model = load_convnext_model(args.model_path, args.model_name, len(class_names), device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
        
    # Evaluate
    plot_path = os.path.dirname(args.model_path)
    if not plot_path: plot_path = '.'
        
    evaluate_model(model, dataloader, class_names, device, plot_path)

if __name__ == "__main__":
    main()
