Detailed Explanation of the Model.

This file describes a deep learning–based flower classification system built using the ConvNeXt architecture, a modern convolutional neural network that blends the strengths of traditional CNNs with design ideas inspired by Transformer models. The primary model used is ConvNeXt Base, while ConvNeXt Large is mentioned as an alternative option for higher capacity when more computational resources are available. The model uses pre-trained ImageNet weights, which allows it to start with rich visual feature representations and then fine-tune those features specifically for flower classification.

The input image size is fixed at 224 × 224 pixels, which is a standard resolution for many ImageNet-trained models. Resizing images to this size ensures compatibility with the pre-trained ConvNeXt architecture and helps maintain a balance between computational efficiency and feature detail.

Dataset Structure and Splitting

The dataset contains a total of 4,317 flower images, distributed across five distinct flower classes:

Rose: 784 images

Sunflower: 733 images

Tulip: 984 images

Daisy: 764 images

Dandelion: 1,052 images

This class distribution shows moderate imbalance, which is important to consider when evaluating performance. To ensure fair training and evaluation, the dataset is split randomly using a fixed seed value of 42, which guarantees reproducibility. The split ratio is:

70% Training set – used to learn model parameters

15% Validation set – used to tune hyperparameters and select the best model

15% Test set – used for final, unbiased performance evaluation

Training and Model Performance

During training, the system continuously monitors training accuracy and validation accuracy at each epoch. The best model checkpoint is saved based on the highest validation accuracy, which peaks at approximately 90.73%. This approach helps prevent overfitting by ensuring that the chosen model generalizes well to unseen data.

The training accuracy reaches around 94.56%, indicating that the model fits the training data very well while still maintaining strong validation performance.

Evaluation Metrics

Beyond accuracy, the model uses more informative classification metrics calculated with a weighted average, which is especially important due to class imbalance:

Precision: 90.23%

Measures how many predicted flower labels are correct.

Recall: 90.78%

Measures how well the model detects all actual flower samples.

F1 Score: 90.64%

The harmonic mean of precision and recall, giving a balanced view of performance.

These metrics together indicate that the model performs consistently well across all five flower categories, not favoring one class at the expense of others.

Overall Significance

In summary, this file documents a well-structured, high-performing image classification pipeline using ConvNeXt and transfer learning. With strong accuracy and balanced precision–recall metrics, the model demonstrates reliable generalization for multi-class flower recognition, making it suitable for academic projects, research demonstrations, or real-world image classification applications 
