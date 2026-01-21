"""
Hyperparameter tuning using Optuna.
"""

import optuna
from optuna.trial import Trial
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""
    
    def __init__(
        self,
        model_name: str,  # ✅ Changed from model_factory
        num_classes: int,  # ✅ Added num_classes
        train_loader,
        val_loader,
        device: str = 'cuda'
    ):
        self.model_name = model_name  # ✅ Store model name
        self.num_classes = num_classes  # ✅ Store num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy
        """
        from Models.models import ModelFactory  # ✅ Import here
        from Models.training import ModelTrainer
        
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        
        # For transformers, suggest lower learning rates
        if 'vit' in self.model_name or 'swin' in self.model_name or 'beit' in self.model_name:
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        
        # Suggest freeze backbone option for pretrained models
        freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])
        
        # Create model using ModelFactory
        model = ModelFactory.create_model(
            model_name=self.model_name,  # ✅ Use stored model_name
            num_classes=self.num_classes,  # ✅ Use stored num_classes
            pretrained=True if self.model_name != 'custom_cnn' else False,
            freeze_backbone=freeze_backbone
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            save_dir=f'temp_optuna_{self.model_name}'
        )
        
        # Train for a few epochs (short tuning)
        try:
            history = trainer.train(
                num_epochs=5,  # Short training for tuning
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                scheduler_type='cosine',
                early_stopping_patience=3
            )
            
            # Return best validation accuracy
            return max(history['val_acc'])
        
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0  # Return poor score if trial fails
    
    def tune(self, n_trials: int = 20) -> Dict:
        """
        Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_name} ({n_trials} trials)...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.model_name}_tuning'
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Hyperparameter Tuning Results for {self.model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation accuracy: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        return study.best_params