"""Training loop for Lewis superadditivity experiment.

Trains connectors and task head while keeping vision models frozen.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .models import ModelBank
from .connectors import ComposedSystem
from .config import ConditionConfig
from .utils import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for one epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float


@dataclass
class TrainingResult:
    """Results from training a condition."""
    best_model_state: Dict[str, Any]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    training_history: List[TrainingMetrics]
    total_train_time: float


def train_condition(
    model_bank: ModelBank,
    condition_config: ConditionConfig, 
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 10,
    patience: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 128
) -> TrainingResult:
    """Train a single experimental condition.
    
    Args:
        model_bank: Frozen vision models
        condition_config: Which models/connectors to use
        train_loader: Training data
        val_loader: Validation data  
        device: Training device
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        learning_rate: AdamW learning rate
        weight_decay: AdamW weight decay
        batch_size: Expected batch size (for logging)
        
    Returns:
        TrainingResult with best model and training history
    """
    start_time = time.time()
    
    logger.info(f"Training condition: {condition_config.name}")
    logger.info(f"Active models: {condition_config.active_models}")
    logger.info(f"Connectors: {condition_config.connector_pairs}")
    
    # Create composed system for this condition
    system = ComposedSystem(
        model_bank=model_bank,
        active_models=condition_config.active_models,
        connector_pairs=condition_config.connector_pairs,
        connector_config=condition_config.connector_config,
        task_head_config=condition_config.task_head_config
    ).to(device)
    
    # Only train connectors and task head
    trainable_params = []
    for name, param in system.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {num_trainable:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=max_epochs,
        eta_min=learning_rate * 0.01
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training state
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    logger.info("Starting training...")
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # Training phase
        system.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, questions, answers) in enumerate(train_loader):
            images = images.to(device)
            questions = questions.to(device) if questions is not None else None
            answers = answers.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = system(images, questions)
            loss = criterion(logits, answers)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, batch {batch_idx}: loss={loss.item():.4f}")
        
        # Validation phase
        system.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, questions, answers in val_loader:
                images = images.to(device)
                questions = questions.to(device) if questions is not None else None
                answers = answers.to(device)
                
                logits = system(images, questions)
                loss = criterion(logits, answers)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Compute accuracy
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == answers).sum().item()
                val_total += answers.size(0)
        
        # Compute epoch metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            learning_rate=current_lr,
            epoch_time=epoch_time
        )
        training_history.append(metrics)
        
        # Print progress
        logger.info(
            f"Epoch {epoch:2d}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"lr={current_lr:.2e}, "
            f"time={epoch_time:.1f}s"
        )
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_state = {
                'system_state_dict': system.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            }
            patience_counter = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    total_train_time = time.time() - start_time
    
    logger.info(
        f"Training completed in {total_train_time:.1f}s. "
        f"Best epoch: {best_epoch}, "
        f"best val_loss: {best_val_loss:.4f}, "
        f"best val_accuracy: {best_val_accuracy:.4f}"
    )
    
    return TrainingResult(
        best_model_state=best_model_state,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        training_history=training_history,
        total_train_time=total_train_time
    )


def load_trained_system(
    model_bank: ModelBank,
    condition_config: ConditionConfig,
    checkpoint_path: str,
    device: torch.device
) -> ComposedSystem:
    """Load a trained system from checkpoint.
    
    Args:
        model_bank: Frozen vision models
        condition_config: System configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Trained ComposedSystem
    """
    # Create system architecture
    system = ComposedSystem(
        model_bank=model_bank,
        active_models=condition_config.active_models,
        connector_pairs=condition_config.connector_pairs,
        connector_config=condition_config.connector_config,
        task_head_config=condition_config.task_head_config
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    system.load_state_dict(checkpoint['system_state_dict'])
    
    logger.info(f"Loaded trained system from {checkpoint_path}")
    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}, val_accuracy={checkpoint['val_accuracy']:.4f}")
    
    return system