"""Trainer class with MLflow integration."""

import os
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import confusion_matrix

from ..utils.metrics import calculate_metrics
from ..utils.logging import setup_logger


class MultimodalTrainer:
    """
    Training orchestrator with MLflow tracking.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        config: Configuration dictionary
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        config: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.logger = setup_logger('MultimodalTrainer')

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # ✅ ALMACENAR MÉTRICAS POR ÉPOCA
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.train_f1s: List[float] = []
        self.train_aucs: List[float] = []
        
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.val_f1s: List[float] = []
        self.val_aucs: List[float] = []
        
        # ✅ ALMACENAR PREDICCIONES FINALES PARA CONFUSION MATRIX
        self.best_val_predictions = None
        self.best_val_labels = None

        mlflow.set_tracking_uri(config.get('tracking_uri', 'sqlite:///mlflow.db'))

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]"
        )
        '''
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            labels = batch['label'].unsqueeze(1)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        '''

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(batch)
            labels = batch['label'].unsqueeze(1)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # ✅ Verificar NaN
            if torch.isnan(loss):
                self.logger.error(f"NaN loss detected at batch {batch_idx}")
                self.logger.error(f"Outputs: {outputs}")
                self.logger.error(f"Labels: {labels}")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            
            # ✅ IMPORTANTE: Actualizar scheduler después de cada batch si es OneCycleLR
            if self.scheduler is not None:
                # Para OneCycleLR, ReduceLROnPlateau requiere step por época
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            # Track metrics
            running_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())

            # ✅ Update progress bar con más info
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': f'{current_lr:.2e}',
                'batch': f'{batch_idx+1}/{len(self.train_loader)}'
            })

        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_predictions, all_labels)

        return {
            'loss': avg_loss,
            **metrics
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]"
        )

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(batch)#LOGITS
            labels = batch['label'].unsqueeze(1)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()


            #all_predictions.append(outputs)
            #all_labels.append(labels)

            # --- CORRECCIÓN ---
            # Aplicar Sigmoid solo para las métricas y guardado
            probs = torch.sigmoid(outputs)  # Convertir Logits -> Probabilidad (0-1)

            # Guardar las PROBABILIDADES, no los logits
            all_predictions.append(probs) 
            all_labels.append(labels)

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate validation metrics
        avg_loss = running_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_predictions, all_labels)

        # ✅ RETORNAR TAMBIÉN LAS PREDICCIONES
        return {
            'loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels,
            **metrics
        }

    def train(self, experiment_name: str = "multimodal_fake_news"):
        """
        Full training loop with MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
        """
        # Setup MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"run_{self.config.get('version', '1.0')}"):
            # Log hyperparameters
            mlflow.log_params({
                'batch_size': self.config.get('batch_size', 32),
                'learning_rate': self.config.get('learning_rate', 1e-4),
                'num_epochs': self.config.get('num_epochs', 10),
                'embedding_dim': self.config.get('embedding_dim', 512),
                'fusion_hidden_dim': self.config.get('fusion_hidden_dim', 256),
                'max_length': self.config.get('max_length', 128),
                'optimizer': self.optimizer.__class__.__name__,
                'scheduler': self.scheduler.__class__.__name__ if self.scheduler else 'None',
                'model_version': self.config.get('version', '1.0'),
                'n_rows': self.config.get('n_rows', '-1'),
            })

            # Log model architecture
            param_counts = self.model.count_parameters()
            mlflow.log_params(param_counts)

            # Log dataset info
            mlflow.log_params({
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'num_batches': len(self.train_loader)
            })
            
            # Log tags para filtrado
            mlflow.set_tags({
                "model_type": "multimodal",
                "fusion_method": self.config.get('fusion_strategy', 'weighted'),
                "stage": "training"
            })

            self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")
            self.logger.info(f"Total parameters: {param_counts['total']:,}")
            self.logger.info(f"Trainable parameters: {param_counts['trainable']:,}")

            for epoch in range(self.config['num_epochs']):
                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate
                val_metrics = self.validate(epoch)

                # ✅ ALMACENAR MÉTRICAS
                self.train_losses.append(train_metrics['loss'])
                self.train_accuracies.append(train_metrics['accuracy'])
                self.train_f1s.append(train_metrics['f1'])
                self.train_aucs.append(train_metrics['auc'])
                
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                self.val_f1s.append(val_metrics['f1'])
                self.val_aucs.append(val_metrics['auc'])

                # Update learning rate
                #if self.scheduler:
                #    self.scheduler.step()

                # ✅ Update scheduler only if NOT is OneCycleLR (updated already by  batch)
                if self.scheduler is not None:
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()

                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'train_auc': train_metrics['auc'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)

                # Print epoch summary
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                    
                    # ✅ GUARDAR PREDICCIONES DEL MEJOR MODELO
                    self.best_val_predictions = val_metrics['predictions'].cpu().numpy()
                    self.best_val_labels = val_metrics['labels'].cpu().numpy()

                    # Save checkpoint
                    checkpoint_path = os.path.join(
                        self.config.get('checkpoint_dir', 'checkpoints'),
                        f'best_model_{self.config.get('version', '1.0')}.pth'
                    )
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'config': self.config
                    }, checkpoint_path)

                    self.logger.info(f"Saved best model with val_acc: {self.best_val_acc:.4f}")

                    # Log model to MLflow
                    #mlflow.pytorch.log_model(self.model, "best_model")
                    #mlflow.pytorch.log_model(self.model, 
                    #    artifact_path="model",  # change to "model" standar
                    #    registered_model_name="MultimodalFakeNews"  # Optional
                    #)
                    signature, input_example = self._create_model_signature()

                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                        conda_env=self._create_conda_env(),
                        #code_paths=["src/"],  # Include source code
                        registered_model_name=None  # Set if you want Model Registry
                    )

            # ✅ GENERAR Y GUARDAR VISUALIZACIONES AL FINAL
            self._log_training_curves()
            self._log_confusion_matrix()

            # Log final best metrics
            mlflow.log_metrics({
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_acc
            })

            self.logger.info("Training completed!")
            self.logger.info(f"Best Val Accuracy: {self.best_val_acc:.4f}")

            return {
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_acc
            }

    def _log_training_curves(self):
        """Genera y guarda gráficas de entrenamiento."""
        if len(self.train_losses) == 0:
            self.logger.warning("No hay métricas para graficar")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Accuracy
        axes[0, 0].plot(epochs, self.train_accuracies, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_accuracies, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.train_f1s, 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_f1s, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('F1 Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(epochs, self.train_aucs, 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.val_aucs, 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title('AUC-ROC', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        model_version= self.config.get('version', '1.0')
        # Guardar en directorio temporal
        curves_path = f'training_curves_{model_version}.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        
        # Log a MLflow
        mlflow.log_artifact(curves_path)
        self.logger.info(f"Training curves guardadas: {curves_path}")
        
        plt.close()

    def _log_confusion_matrix(self):
        """Guarda matriz de confusión del mejor modelo."""
        if self.best_val_predictions is None or self.best_val_labels is None:
            self.logger.warning("No hay predicciones para matriz de confusión")
            return
        
        # Convertir probabilidades a clases (umbral 0.5)
        predictions_binary = (self.best_val_predictions > 0.5).astype(int).flatten()
        labels_binary = self.best_val_labels.astype(int).flatten()
        
        # Calcular matriz de confusión
        cm = confusion_matrix(labels_binary, predictions_binary)
        
        # Graficar
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Best Model', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Añadir métricas en el gráfico
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        plt.text(
            0.5, -0.15, 
            f'Accuracy: {accuracy:.4f}',
            ha='center',
            transform=plt.gca().transAxes,
            fontsize=10
        )
        
        plt.tight_layout()
        
        model_version= self.config.get('version', '1.0')
        # Guardar
        cm_path = f'confusion_matrix_{model_version}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        
        # Log a MLflow
        mlflow.log_artifact(cm_path)
        self.logger.info(f"Confusion matrix guardada: {cm_path}")
        
        plt.close()


    def _create_model_signature(self):
        """Crea signature e input example para MLflow."""
        try:
            from mlflow.models.signature import infer_signature
            import numpy as np
            
            # Obtener un batch de ejemplo
            sample_batch = next(iter(self.val_loader))
            
            # Preparar input (sin labels)
            sample_input = {}
            input_example = {}
            
            for k, v in sample_batch.items():
                if k != 'label' and isinstance(v, torch.Tensor):
                    sample_input[k] = v.to(self.device)
                    # Para input_example, usar solo el primer elemento
                    input_example[k] = v[0:1].cpu().numpy()
            
            # Hacer predicción
            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(sample_batch)
            
            # Crear signature
            signature = infer_signature(
                input_example,
                sample_output[0:1].cpu().numpy()
            )
            
            return signature, input_example
            
        except Exception as e:
            self.logger.warning(f"No se pudo crear model signature: {e}")
            return None, None


    # Crear requirements.txt explícito para MLflow
    def _create_conda_env(self):
        """Crea environment conda para MLflow."""
        import mlflow
        
        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.10',
                {
                    'pip': [
                        f'torch=={torch.__version__}',
                        f'transformers==4.36.0',
                        f'numpy==1.24.3',
                        f'scikit-learn==1.3.2',
                        'Pillow==10.1.0',
                    ]
                }
            ],
            'name': 'multimodal_env'
        }
        return conda_env

        
    def _create_model_signature__old(self):
        """Crea signature para MLflow Model Registry."""
        try:
            from mlflow.models.signature import infer_signature
            
            # Obtener un batch de ejemplo
            sample_batch = next(iter(self.val_loader))
            sample_input = {
                k: v.to(self.device) 
                for k, v in sample_batch.items() 
                if isinstance(v, torch.Tensor) and k != 'label'
            }
            
            # Hacer predicción
            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(sample_batch)
            
            # Crear signature
            signature = infer_signature(
                sample_input,
                sample_output.cpu().numpy()
            )
            
            return signature
            
        except Exception as e:
            self.logger.warning(f"No se pudo crear model signature: {e}")
            return None