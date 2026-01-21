"""
Main script to train and compare all models.
"""

import torch
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import pandas as pd

from Models.data_loader import get_data_loaders, get_class_names
from Models.models import ModelFactory, EnsembleModel
from Models.training import ModelTrainer
from Models.hyperparam_tuning import HyperparameterTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare multiple models and select the best."""
    
    def __init__(
        self,
        data_dir: str,
        save_dir: str = 'experiments',
        device: str = None,
        use_hyperparameter_tuning: bool = True,
        tuning_trials: int = 20
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.tuning_trials = tuning_trials
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            str(self.data_dir),
            batch_size=32,
            num_workers=0 # has to be 0 for windows otherwise gets frozen
        )
        
        self.class_names = get_class_names(str(self.data_dir))
        self.num_classes = len(self.class_names)
        
        self.results = {}
        self.best_hyperparams = {}
    
    def tune_hyperparameters(self, model_name: str, n_trials: int = None) -> Dict:
        """
        Tune hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            n_trials: Number of tuning trials (uses default if None)
            
        Returns:
            Best hyperparameters
        """
        if n_trials is None:
            n_trials = self.tuning_trials
        
        tuner = HyperparameterTuner(
            model_name=model_name,
            num_classes=self.num_classes,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        best_params = tuner.tune(n_trials=n_trials)
        
        # Save hyperparameters
        hyperparam_dir = self.save_dir / 'hyperparameters'
        hyperparam_dir.mkdir(exist_ok=True)
        
        with open(hyperparam_dir / f'{model_name}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        self.best_hyperparams[model_name] = best_params
        
        return best_params
    
    def train_model(
        self,
        model_name: str,
        num_epochs: int = 30,
        learning_rate: float = None,
        pretrained: bool = True,
        freeze_backbone: bool = None,
        optimizer_type: str = 'adamw',
        tune_first: bool = None
    ) -> Dict:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            num_epochs: Number of training epochs
            learning_rate: Learning rate (tuned if None and tune_first=True)
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone layers (tuned if None and tune_first=True)
            optimizer_type: Optimizer type
            tune_first: Whether to tune hyperparameters first (uses class default if None)
            
        Returns:
            Evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")
        
        # Determine if we should tune
        if tune_first is None:
            tune_first = self.use_hyperparameter_tuning
        
        # Tune hyperparameters if requested
        if tune_first:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            best_params = self.tune_hyperparameters(model_name)
            
            # Use tuned parameters
            learning_rate = best_params.get('learning_rate', learning_rate or 1e-3)
            optimizer_type = best_params.get('optimizer', optimizer_type)
            freeze_backbone = best_params.get('freeze_backbone', freeze_backbone or False)
        else:
            # Use provided or default values
            learning_rate = learning_rate or 1e-3
            freeze_backbone = freeze_backbone or False
        
        logger.info(f"Using hyperparameters:")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Optimizer: {optimizer_type}")
        logger.info(f"  Freeze backbone: {freeze_backbone}")
        
        # Create model
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=self.num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Create save directory for this model
        model_save_dir = self.save_dir / model_name
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            save_dir=str(model_save_dir)
        )
        
        # Train
        history = trainer.train(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            scheduler_type='cosine',
            early_stopping_patience=10
        )
        
        # Plot training history
        trainer.plot_training_history(
            save_path=str(model_save_dir / 'training_history.png')
        )
        
        # Evaluate on test set
        results = trainer.evaluate(self.test_loader, self.class_names)
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(
            cm=results['confusion_matrix'],
            class_names=self.class_names,
            save_path=str(model_save_dir / 'confusion_matrix.png')
        )
        
        # Save results
        results['model_name'] = model_name
        results['history'] = history
        results['hyperparameters'] = {
            'learning_rate': learning_rate,
            'optimizer': optimizer_type,
            'freeze_backbone': freeze_backbone,
            'num_epochs': num_epochs
        }
        
        with open(model_save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results[model_name] = results
        
        return results
    
    def train_all_models(self, tune_hyperparameters: bool = None) -> Dict:
        """
        Train all models in the comparison.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters (uses class default if None)
            
        Returns:
            Dictionary of all results
        """
        if tune_hyperparameters is None:
            tune_hyperparameters = self.use_hyperparameter_tuning
        
        models_to_train = [
            # Pre-trained CNNs (faster to train)
            ('resnet50', {'num_epochs': 30}),
            ('efficientnet_b0', {'num_epochs': 30}),
            
            # Plant-specific model
            ('plant_classifier', {'num_epochs': 30}),
            
            # Custom CNN
            # ('custom_cnn', {'num_epochs': 50, 'pretrained': False}),
            
            # Vision Transformer (pick one - they're slow)
            # ('vit_base', {'num_epochs': 30}),
            
            # Hybrid model (efficient)
            ('convnext_tiny', {'num_epochs': 30}),
        ]
        
        for model_name, params in models_to_train:
            try:
                self.train_model(
                    model_name, 
                    tune_first=tune_hyperparameters,
                    **params
                )
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.results
    
    def create_ensemble(self, model_names: List[str]) -> Dict:
        """
        Create and evaluate an ensemble of models.
        
        Args:
            model_names: List of model names to ensemble
            
        Returns:
            Ensemble evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating Ensemble: {', '.join(model_names)}")
        logger.info(f"{'='*60}")
        
        models = []
        for model_name in model_names:
            model = ModelFactory.create_model(
                model_name=model_name,
                num_classes=self.num_classes,
                pretrained=False
            )
            
            # Load trained weights
            checkpoint_path = self.save_dir / model_name / 'best_model.pth'
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found for {model_name}, skipping...")
                continue
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            models.append(model)
        
        if len(models) < 2:
            logger.error("Need at least 2 models for ensemble")
            return {}
        
        # Create ensemble
        ensemble = EnsembleModel(models)
        
        # Evaluate
        trainer = ModelTrainer(
            model=ensemble,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            save_dir=str(self.save_dir / 'ensemble')
        )
        
        results = trainer.evaluate(self.test_loader, self.class_names)
        results['model_name'] = f"Ensemble({', '.join(model_names)})"
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(
            cm=results['confusion_matrix'],
            class_names=self.class_names,
            save_path=str(self.save_dir / 'ensemble' / 'confusion_matrix.png')
        )
        
        # Save results
        with open(self.save_dir / 'ensemble' / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results['ensemble'] = results
        
        return results
    
    def compare_results(self) -> pd.DataFrame:
        """Compare all model results."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        # Save comparison
        df.to_csv(self.save_dir / 'model_comparison.csv', index=False)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"\n{df.to_string(index=False)}")
        
        return df


def main():
    """Main execution function."""
    
    # Configuration
    data_dir = r"d:\OneDrive\PlantDoctor\datasets\processed"
    save_dir = r"d:\OneDrive\PlantDoctor\experiments"
    
    # Create comparison with hyperparameter tuning
    comparison = ModelComparison(
        data_dir=data_dir,
        save_dir=save_dir,
        use_hyperparameter_tuning=False,  # Enable hyperparameter tuning
        tuning_trials=15  # Number of trials per model
    )
    
    # Train all models (will tune hyperparameters first)
    results = comparison.train_all_models(tune_hyperparameters=False)
    # Just verify the code works - won't produce good model
    # results = comparison.train_model(
    #     'mobilenet_v3_small',  # Faster model
    #     num_epochs=1,  # Just 1 epoch to test
    #     tune_first=False
    # )
    # # Create ensemble of top 3 models
    if len(results) >= 3:
        top_models = sorted(
            [(name, res['accuracy']) for name, res in results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        top_model_names = [name for name, _ in top_models]
        logger.info(f"\nTop 3 models: {top_model_names}")
        
        comparison.create_ensemble(top_model_names)
    
    # Final comparison
    comparison_df = comparison.compare_results()
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()