import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, version='121'):
        """
        DenseNet model for breast cancer classification
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            version (str): DenseNet version (121, 169, 201)
        """
        super(DenseNetModel, self).__init__()
        
        # Choose DenseNet version
        if version == '121':
            base_model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = base_model.classifier.in_features
        elif version == '169':
            base_model = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = base_model.classifier.in_features
        else:
            raise ValueError(f"Unsupported DenseNet version: {version}")
            
        # Replace classifier
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
            
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ResNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, version='50'):
        """
        ResNet model for breast cancer classification
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            version (str): ResNet version (18, 50, 101)
        """
        super(ResNetModel, self).__init__()
        
        # Choose ResNet version
        if version == '18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif version == '50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif version == '101':
            base_model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet version: {version}")
            
        # Remove the last fully connected layer
        num_ftrs = base_model.fc.in_features
        
        # Create new layers
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
            
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, version='b0'):
        """
        EfficientNet model for breast cancer classification
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            version (str): EfficientNet version (b0, b3)
        """
        super(EfficientNetModel, self).__init__()
        
        # Choose EfficientNet version
        if version == 'b0':
            base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = base_model.classifier[1].in_features
        elif version == 'b3':
            base_model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = base_model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported EfficientNet version: {version}")
            
        # Replace classifier
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
            
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes=2, pretrained=True, version=None):
    """
    Factory function to create a model
    
    Args:
        model_name (str): Model name ('densenet', 'resnet', 'efficientnet')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        version (str): Model version
        
    Returns:
        model: PyTorch model
    """
    if model_name.lower() == 'densenet':
        version = '121' if version is None else version
        return DenseNetModel(num_classes=num_classes, pretrained=pretrained, version=version)
    elif model_name.lower() == 'resnet':
        version = '50' if version is None else version
        return ResNetModel(num_classes=num_classes, pretrained=pretrained, version=version)
    elif model_name.lower() == 'efficientnet':
        version = 'b0' if version is None else version
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained, version=version)
    else:
        raise ValueError(f"Unsupported model: {model_name}") 