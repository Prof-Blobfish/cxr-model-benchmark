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
    def __init__(self, num_classes=2, pretrained=True, in_channels=1):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.conv1 = new_conv

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.features.conv0
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.features.conv0 = new_conv

        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.features[0][0] = new_conv

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.features[0][0] = new_conv

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.conv1[0]
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=False
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.conv1[0] = new_conv

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.features[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            self.model.features[0] = new_conv

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