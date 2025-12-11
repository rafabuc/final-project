"""Main training script with Hydra configuration."""
import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import MultimodalDataset
from src.data.transforms import get_image_transforms
from src.models.multimodal_net import MultimodalNet
from src.training.trainer import MultimodalTrainer
from src.utils.logging import setup_logger
from torch.optim.lr_scheduler import OneCycleLR

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup logger
    logger = setup_logger('TrainingPipeline', log_file='training.log')
    logger.info("Starting Multimodal Fake News Detection Training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)

    # Initialize tokenizer
    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # --- CORRECCIÓN AQUÍ: CARGAR DATASETS POR SEPARADO ---
    logger.info("Loading datasets from separate CSV splits...")

    # 1. TRAIN DATASET (Con Data Augmentation)
    # Asumimos que en tu config.yaml tienes cfg.data.train_file y cfg.data.val_file
    train_dataset = MultimodalDataset(
        csv_file=os.path.join(cfg.data.root_dir, 'train.csv'), # O cfg.data.train_file
        root_dir=cfg.data.images_dir_train,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        # IMPORTANTE: Mode='train' aplica rotaciones y ruido
        image_transform=get_image_transforms(mode='train', image_size=cfg.data.image_size),
        text_column=cfg.data.text_column,
        image_column=cfg.data.image_column,
        label_column=cfg.data.label_column,
        n_rows=cfg.data.n_rows
    )

    # 2. VALIDATION DATASET (Limpio, sin Augmentation)
    val_dataset = MultimodalDataset(
        csv_file=os.path.join(cfg.data.root_dir, 'val.csv'), # O cfg.data.val_file
        root_dir=cfg.data.images_dir_val,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        # IMPORTANTE: Mode='val' solo hace resize y normalización
        image_transform=get_image_transforms(mode='val', image_size=cfg.data.image_size),
        text_column=cfg.data.text_column,
        image_column=cfg.data.image_column,
        label_column=cfg.data.label_column,
        n_rows=cfg.data.n_rows
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True, # Shuffle SIEMPRE True en Train
        num_workers=cfg.training.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False, # Shuffle False en Val
        num_workers=cfg.training.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    logger.info("Initializing MultimodalNet...")
    model = MultimodalNet(
        embedding_dim=cfg.model.embedding_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        dropout_rate=cfg.model.dropout_rate,
        # Asegúrate de pasar estos params en tu config.yaml
        freeze_vision=cfg.model.get('freeze_vision', False),
        freeze_text=cfg.model.get('freeze_text', False) 
    )

    #logger.info(f'state====================={model.state_dict()}')
    load_model = cfg.model.get('load_model', False)
        
    if load_model: 
        # ---Load previous weights (Fase 1) ---
        best_model_name = cfg.model.get('best_model_name', 'best_model.pth')
        checkpoint_path = os.path.join(cfg.training.checkpoint_dir, best_model_name)
        logger.info(f"Try Loading weights from {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load state_dict (manejando posibles diferencias de prefijos)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Weights loaded successfully 1")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Weights loaded successfully 2")
            
            #logger.info(f'state====================={len(model.state_dict())}')
            #raise ValueError(f"FOUND correctly checkpoint  {checkpoint_path}")    
        else:

            logger.warning("¡Not found checkpoint! training from scratch (Dangerous if freeze=False)")
            raise ValueError(f"Not found checkpoint  {checkpoint_path}")    

    model = model.to(device)
    
    # Optimizer
    if cfg.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    elif cfg.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    

    # Learning rate scheduler
    scheduler = None
    if cfg.training.use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.training.learning_rate,#2e-4,
            epochs=cfg.training.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        '''
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_epochs,
            eta_min=cfg.training.min_lr
        )
        '''

    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # Create trainer
    trainer_config = {
        'num_epochs': cfg.training.num_epochs,
        'batch_size': cfg.training.batch_size,
        'learning_rate': cfg.training.learning_rate,
        'embedding_dim': cfg.model.embedding_dim,
        'fusion_hidden_dim': cfg.model.fusion_hidden_dim,
        'max_length': cfg.data.max_length,
        'checkpoint_dir': cfg.training.checkpoint_dir,
        'version': cfg.training.version,
        'tracking_uri': cfg.training.tracking_uri,
        'n_rows': cfg.data.n_rows      
    }

    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=trainer_config,
        scheduler=scheduler
    )

    logger.info("Starting training loop...")
    trainer.train(experiment_name=cfg.training.experiment_name)

if __name__ == "__main__":
    main()