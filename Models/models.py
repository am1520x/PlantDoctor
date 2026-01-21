"""
Model architectures for plant classification.
Includes pre-trained models, custom CNNs, transformers, and hybrid architectures.
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from typing import Optional, List


class ModelFactory:
    """Factory class to create different model architectures."""
    
    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ) -> nn.Module:
        """
        Create a model by name.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers
            
        Returns:
            PyTorch model
        """
        
        # ResNet models
        if model_name == 'resnet50':
            return ModelFactory._create_resnet50(num_classes, pretrained, freeze_backbone)
        elif model_name == 'resnet101':
            return ModelFactory._create_resnet101(num_classes, pretrained, freeze_backbone)
        
        # EfficientNet models
        elif model_name.startswith('efficientnet'):
            return ModelFactory._create_efficientnet(model_name, num_classes, pretrained, freeze_backbone)
        
        # Vision Transformer models
        elif model_name == 'vit_base':
            return ModelFactory._create_vit_base(num_classes, pretrained, freeze_backbone)
        elif model_name == 'vit_large':
            return ModelFactory._create_vit_large(num_classes, pretrained, freeze_backbone)
        
        # Swin Transformer
        elif model_name.startswith('swin'):
            return ModelFactory._create_swin(model_name, num_classes, pretrained, freeze_backbone)
        
        # ConvNeXt
        elif model_name.startswith('convnext'):
            return ModelFactory._create_convnext(model_name, num_classes, pretrained, freeze_backbone)
        
        # BEiT
        elif model_name.startswith('beit'):
            return ModelFactory._create_beit(model_name, num_classes, pretrained, freeze_backbone)
        
        # CoaT (Co-Scale Conv-Attentional Image Transformers)
        elif model_name.startswith('coat'):
            return ModelFactory._create_coat(model_name, num_classes, pretrained, freeze_backbone)
        
        # Custom CNN
        elif model_name == 'custom_cnn':
            return CustomCNN(num_classes)
        
        # Plant-specific pre-trained models (from timm)
        elif model_name == 'plant_classifier':
            return ModelFactory._create_plant_specific(num_classes, pretrained, freeze_backbone)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def _create_resnet50(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create ResNet-50 model."""
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model
    
    @staticmethod
    def _create_resnet101(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create ResNet-101 model."""
        if pretrained:
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet101(weights=None)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model
    
    @staticmethod
    def _create_efficientnet(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create EfficientNet model using timm."""
        # Map to timm model names
        timm_name_map = {
            'efficientnet_b0': 'efficientnet_b0',
            'efficientnet_b1': 'efficientnet_b1',
            'efficientnet_b2': 'efficientnet_b2',
            'efficientnet_b3': 'efficientnet_b3',
            'efficientnet_b4': 'efficientnet_b4',
        }
        
        timm_name = timm_name_map.get(model_name, 'efficientnet_b0')
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            # Freeze all except classifier
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_vit_base(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create Vision Transformer Base model."""
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_vit_large(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create Vision Transformer Large model."""
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_swin(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create Swin Transformer model."""
        timm_name_map = {
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224',
        }
        
        timm_name = timm_name_map.get(model_name, 'swin_tiny_patch4_window7_224')
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_convnext(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create ConvNeXt model."""
        timm_name_map = {
            'convnext_tiny': 'convnext_tiny',
            'convnext_small': 'convnext_small',
            'convnext_base': 'convnext_base',
        }
        
        timm_name = timm_name_map.get(model_name, 'convnext_tiny')
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_beit(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create BEiT model."""
        timm_name_map = {
            'beit_base': 'beit_base_patch16_224',
            'beit_large': 'beit_large_patch16_224',
        }
        
        timm_name = timm_name_map.get(model_name, 'beit_base_patch16_224')
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_coat(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """Create CoaT model."""
        timm_name_map = {
            'coat_tiny': 'coat_tiny',
            'coat_mini': 'coat_mini',
        }
        
        timm_name = timm_name_map.get(model_name, 'coat_tiny')
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        return model
    
    @staticmethod
    def _create_plant_specific(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
        """
        Create a plant-specific pre-trained model.
        Using a model that performs well on plant/botanical images.
        """
        # EfficientNet-B4 trained on iNaturalist (good for biological images)
        model = timm.create_model('tf_efficientnet_b4.ns_jft_in1k', pretrained=pretrained, num_classes=num_classes)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        
        return model


class CustomCNN(nn.Module):
    """Custom CNN architecture for plant classification."""
    
    def __init__(self, num_classes: int):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv_block1 = self._make_conv_block(3, 64)
        self.conv_block2 = self._make_conv_block(64, 128)
        self.conv_block3 = self._make_conv_block(128, 256)
        self.conv_block4 = self._make_conv_block(256, 512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class EnsembleModel(nn.Module):
    """Ensemble multiple models with weighted voting."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs and apply weights
        stacked = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        
        # Weighted average
        weights = self.weights.view(-1, 1, 1).to(x.device)
        weighted_output = (stacked * weights).sum(dim=0)
        
        return weighted_output