import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class MultiLabelResNet50(nn.Module):
    """
    ResNet50-based multi-label classification model.
    Uses pretrained ImageNet weights and replaces the final layer for 4-label classification.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Args:
            num_classes (int): Number of output classes (labels)
            pretrained (bool): Whether to use pretrained ImageNet weights
        """
        super(MultiLabelResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Get the number of features from the last layer
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Initialize the new layer
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class MultiLabelMobileNetV2(nn.Module):
    """
    MobileNetV2-based multi-label classification model.
    Uses pretrained ImageNet weights and replaces the final layer for 4-label classification.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Args:
            num_classes (int): Number of output classes (labels)
            pretrained (bool): Whether to use pretrained ImageNet weights
        """
        super(MultiLabelMobileNetV2, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final fully connected layer
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Initialize the new layer
        nn.init.xavier_uniform_(self.backbone.classifier[1].weight)
        nn.init.zeros_(self.backbone.classifier[1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


def get_model(model_name: str = 'resnet50', num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create a multi-label classification model.
    
    Args:
        model_name (str): Name of the model architecture ('resnet50' or 'mobilenetv2')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        nn.Module: The specified model
    """
    if model_name.lower() == 'resnet50':
        return MultiLabelResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'mobilenetv2':
        return MultiLabelMobileNetV2(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'resnet50' or 'mobilenetv2'")


def save_model(model: nn.Module, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, 
               epoch: Optional[int] = None, loss: Optional[float] = None):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        filepath (str): Path to save the model
        optimizer (Optional[torch.optim.Optimizer]): Optimizer state
        epoch (Optional[int]): Current epoch
        loss (Optional[float]): Current loss
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to the saved model
        model (nn.Module): Model to load state into
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into
    
    Returns:
        dict: Checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0)
    }
    
    print(f"Model loaded from {filepath}")
    return info
