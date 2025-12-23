import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import argparse

class ConvNeXtPredictor:
    def __init__(self, model_path, model_name='convnext_base', class_names_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model, self.class_names = self._load_model_and_classes(model_path, class_names_path)

    def _load_model_and_classes(self, path, class_names_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        print(f"Loading {self.model_name} from {path}...")
        
        if self.model_name == 'convnext_base':
            model = models.convnext_base(weights=None)
        elif self.model_name == 'convnext_large':
            model = models.convnext_large(weights=None)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        elif class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            raise ValueError("Class names not found in checkpoint and no file provided.")

        num_classes = len(class_names)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully with {num_classes} classes")
        
        return model, class_names

    def predict(self, image_path, top_k=5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
            
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(self.class_names)))
            
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'class': self.class_names[idx.item()],
                'confidence': prob.item() * 100
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict using ConvNeXt')
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('--model_path', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--model_name', default='convnext_base', choices=['convnext_base', 'convnext_large'])
    parser.add_argument('--classes', help='Path to class names txt (optional if in checkpoint)')
    parser.add_argument('--top_k', type=int, default=5)
    
    args = parser.parse_args()
    
    try:
        predictor = ConvNeXtPredictor(args.model_path, args.model_name, args.classes)
        results = predictor.predict(args.image_path, args.top_k)
        
        print("\n" + "="*50)
        print(f"Prediction for: {os.path.basename(args.image_path)}")
        print("="*50)
        
        for i, res in enumerate(results, 1):
             print(f"{i}. {res['class']:20s}: {res['confidence']:.2f}%")
             
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()