Model Architecture

Model Used: ConvNeXt Base (with ConvNeXt Large as an alternative option)
Pre-trained weights from ImageNet, fine-tuned for flower classification
Input size: 224×224 pixels

Dataset Split

Training: 70%
Validation: 15%
Test: 15%
Random split with seed=42

Key Metrics (From training script structure)
The model tracks:

Validation Accuracy: Best model saved based on highest val accuracy around 90.73%
Training Accuracy: Monitored per epoch around 94.56%
Precision, Recall, F1 Score: Calculated using weighted average
Precision : 90.23%
Recall : 90.78%
F1 Score : 90.64%

The model consists of five flower classification classes.

Total images used in this dataset are : 4317 images 
Images per class of dataset are
Rose : 784
Sunflower : 733
Tulip : 984
Daisy : 764
Dandelion : 1052
