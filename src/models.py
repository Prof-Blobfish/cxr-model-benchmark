import torch
import torch.nn as nn
from torchvision import models

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

class VGG11(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)

        if in_channels == 1:
            old_conv = self.model.features[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()
            self.model.features[0] = new_conv

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        self.model = models.googlenet(
            weights=models.GoogLeNet_Weights.DEFAULT,
            aux_logits=True,
        )
        # Keep pretrained weight loading compatible, then disable aux outputs
        # so the training loop receives a single logits tensor.
        self.model.aux_logits = False
        self.model.aux1 = None
        self.model.aux2 = None
        if in_channels == 1:
            # Pretrained GoogLeNet enables an RGB-specific transform_input path.
            # Disable it for grayscale tensors to avoid channel index errors.
            self.model.transform_input = False

        if in_channels == 1:
            old_conv = self.model.conv1.conv
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()
            self.model.conv1.conv = new_conv

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)