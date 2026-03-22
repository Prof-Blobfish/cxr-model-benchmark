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

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, in_channels=1):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
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

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.densenet121(weights=None)

        # Modify first conv layer for grayscale
        self.model.features.conv0 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        # Load EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)

        # Modify first conv layer for grayscale input
        self.model.features[0][0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        # Load MobileNetV2
        self.model = models.mobilenet_v2(weights=None)

        # Modify first convolution layer for grayscale input
        self.model.features[0][0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.shufflenet_v2_x1_0(weights=None)

        self.model.conv1[0] = nn.Conv2d(
            in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        # Load SqueezeNet
        self.model = models.squeezenet1_0(weights=None)

        # Modify first conv layer for grayscale input
        self.model.features[0] = nn.Conv2d(
            in_channels,
            96,
            kernel_size=7,
            stride=2
        )

        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return torch.flatten(x, 1)