"""
Training pipeline for plant classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate deep learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(
        self,
        num_epochs: int,
        learning_rate: float = 1e-3,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = 'cosine',
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler_type: Learning rate scheduler ('cosine', 'step', None)
            early_stopping_patience: Patience for early stopping
            save_best_only: Only save the best model
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup learning rate scheduler
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_best_only:
                    self.save_checkpoint('best_model.pth', epoch, optimizer, val_acc)
                    logger.info(f"New best model saved with val_acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training complete! Best val_acc: {best_val_acc:.4f}")
        return self.history
    
    def _train_epoch(self, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader: DataLoader, class_names: list) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            class_names: List of class names
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        logger.info("Evaluating on test set...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_metrics': {
                class_names[i]: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1_score': per_class_f1[i],
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        return results
    
    def save_checkpoint(self, filename: str, epoch: int, optimizer, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()