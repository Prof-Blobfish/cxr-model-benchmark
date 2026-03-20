import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name):
    if model_name == "resnet18":
        return ResNet18(num_classes=2, pretrained=True, in_channels=1)
    elif model_name == "simple_cnn":
        return SimpleCNN()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    # Factory function to create and return a model instance based on the specified model name, allowing for easy switching between different architectures (e.g., ResNet18 or a simple custom CNN)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, in_channels=1):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = models.resnet18(weights=weights)

        if in_channels == 1:
            self.model.conv1 = nn.Conv2d(
                in_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding = 1), # First convolutional layer: input channels = 1 (grayscale), output channels = 16, kernel size = 3x3, padding = 1 to maintain spatial dimensions
            nn.ReLU(), # ReLU activation function for non-linearity
            nn.MaxPool2d(2), # Max pooling layer to downsample the feature maps by a factor of 2

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)

            # Three convolutional blocks, each consisting of a convolutional layer, ReLU activation, and max pooling to progressively extract features and reduce spatial dimensions
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        # Classifier head: flattens the output from the convolutional layers, applies a fully connected layer with 128 units, ReLU activation, dropout for regularization, and a final output layer with 2 units for binary classification

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    # Defines the forward pass of the model, passing the input through the feature extractor and then the classifier to produce the final output logits