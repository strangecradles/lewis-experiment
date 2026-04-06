#!/usr/bin/env python3
"""Main orchestration script for Lewis superadditivity experiment.

Loads all models once, then trains and evaluates all conditions.
Designed for GH200: everything stays in memory, no reloading.
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp

from lewis.models import ModelBank
from lewis.dataset import create_gqa_dataloaders
from lewis.config import get_all_conditions
from lewis.train import train_condition, TrainingResult
from lewis.evaluate import evaluate_condition, compute_all_metrics, EvaluationResult
from lewis.utils import setup_logging, set_random_seed, get_device_info, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete Lewis superadditivity experiment"
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use (cuda, cpu, or auto)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate pipeline without training'
    )
    
    parser.add_argument(
        '--num-train',
        type=int,
        default=None,
        help='Limit number of training samples (for testing)'
    )
    
    parser.add_argument(
        '--num-eval', 
        type=int,
        default=None,
        help='Limit number of evaluation samples (for testing)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--max-epochs',
        type=int, 
        default=10,
        help='Maximum training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--conditions',
        type=str,
        nargs='*',
        default=None,
        help='Specific conditions to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true', 
        help='Skip training, only do evaluation (assumes checkpoints exist)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device with memory info."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger = get_logger(__name__)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        device_info = get_device_info()
        logger.info(f"GPU: {device_info.get('name', 'unknown')}")
        logger.info(f"Memory: {device_info.get('memory_gb', 0):.1f} GB")
        
        # Enable memory efficiency settings for large models
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device


def save_results(
    results_dir: Path,
    condition_name: str,
    training_result: Optional[TrainingResult],
    evaluation_result: EvaluationResult,
    args: argparse.Namespace
):
    """Save results for a single condition."""
    condition_dir = results_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    eval_data = {
        'condition_name': evaluation_result.condition_name,
        'overall_accuracy': evaluation_result.overall_accuracy,
        'accuracy_by_question_type': evaluation_result.accuracy_by_question_type,
        'total_questions': evaluation_result.total_questions,
        'questions_by_type': evaluation_result.questions_by_type,
        'eval_time': evaluation_result.eval_time
    }
    
    with open(condition_dir / 'evaluation.json', 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    # Save training results if available
    if training_result is not None:
        train_data = {
            'best_epoch': training_result.best_epoch,
            'best_val_loss': training_result.best_val_loss,
            'best_val_accuracy': training_result.best_val_accuracy,
            'total_train_time': training_result.total_train_time,
            'training_history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'val_accuracy': m.val_accuracy,
                    'learning_rate': m.learning_rate,
                    'epoch_time': m.epoch_time
                }
                for m in training_result.training_history
            ]
        }
        
        with open(condition_dir / 'training.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Save model checkpoint
        if training_result.best_model_state is not None:
            torch.save(
                training_result.best_model_state,
                condition_dir / 'best_model.pth'
            )
    
    # Save experiment configuration
    config_data = {
        'args': vars(args),
        'condition_name': condition_name
    }
    
    with open(condition_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)


def print_summary_table(results: Dict[str, EvaluationResult]):
    """Print a summary table of all results."""
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    # Sort results by condition name for consistent ordering
    sorted_results = sorted(results.items())
    
    # Print table header
    header = f"{'Condition':<20} {'Accuracy':<10} {'Questions':<10} {'Time (s)':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print results
    for condition_name, result in sorted_results:
        row = f"{condition_name:<20} {result.overall_accuracy:<10.4f} {result.total_questions:<10} {result.eval_time:<10.1f}"
        logger.info(row)
    
    # Print question type breakdown for key conditions
    key_conditions = ['A_alone', 'B_alone', 'C_alone', 'A+B+C']
    for condition_name in key_conditions:
        if condition_name in results:
            result = results[condition_name]
            if result.accuracy_by_question_type:
                logger.info(f"\n{condition_name} by question type:")
                for qtype, acc in result.accuracy_by_question_type.items():
                    count = result.questions_by_type.get(qtype, 0)
                    logger.info(f"  {qtype}: {acc:.4f} ({count} questions)")


def main():
    """Main experiment orchestration."""
    args = parse_args()
    
    # Setup
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    logger = get_logger(__name__)
    
    logger.info("Starting Lewis superadditivity experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata
    with open(results_dir / 'experiment_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # Load data once
        logger.info("Loading GQA dataset...")
        data_start = time.time()
        train_loader, val_loader, answer_vocab = create_gqa_dataloaders(
            batch_size=args.batch_size,
            train_samples=args.num_train,
            eval_samples=args.num_eval
        )
        data_time = time.time() - data_start
        logger.info(f"Data loaded in {data_time:.1f}s")
        
        # Load all models once
        logger.info("Loading vision models...")
        model_start = time.time()
        model_bank = ModelBank(device=device)
        model_time = time.time() - model_start
        logger.info(f"Models loaded in {model_time:.1f}s")
        
        # Get experimental conditions
        all_conditions = get_all_conditions()
        if args.conditions:
            # Filter to specific conditions
            conditions = [c for c in all_conditions if c.condition_name in args.conditions]
            if not conditions:
                raise ValueError(f"No matching conditions found for: {args.conditions}")
        else:
            conditions = all_conditions
        
        logger.info(f"Running {len(conditions)} conditions")
        
        if args.dry_run:
            logger.info("Dry run mode - validating pipeline without training")
            for condition in conditions[:2]:  # Just test first 2 conditions
                logger.info(f"Would train condition: {condition.condition_name}")
            return
        
        # Run all conditions
        all_results = {}
        total_start = time.time()
        
        num_classes = len(answer_vocab)
        logger.info(f"Number of answer classes: {num_classes}")
        
        from lewis.connectors import ComposedSystem, ConnectorConfig
        
        for i, condition in enumerate(conditions):
            logger.info(f"\n{'-'*60}")
            logger.info(f"Condition {i+1}/{len(conditions)}: {condition.condition_name}")
            logger.info(f"{'-'*60}")
            
            condition_start = time.time()
            active_models = sorted(list(condition.model_subset.models))
            logger.info(f"Active models: {active_models}")
            logger.info(f"Connector type: {condition.connector_type}")
            
            # Set seed for this condition
            set_random_seed(condition.seed)
            
            training_result = None
            
            # Training phase
            if not args.skip_training:
                logger.info("Starting training...")
                training_result = train_condition(
                    model_bank=model_bank,
                    condition_config=condition,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                    max_epochs=args.max_epochs,
                    learning_rate=args.learning_rate
                )
                logger.info(f"Training completed in {training_result.total_train_time:.1f}s")
            
            # Evaluation phase  
            logger.info("Starting evaluation...")
            
            # Create system for evaluation
            system = ComposedSystem(
                model_bank=model_bank,
                active_models=active_models,
                num_classes=num_classes,
                connector_config=ConnectorConfig()
            ).to(device)
            
            # Load trained weights if available
            if training_result is not None and training_result.best_model_state is not None:
                system.load_state_dict(training_result.best_model_state['system_state_dict'])
            
            evaluation_result = evaluate_condition(
                system=system,
                val_loader=val_loader,
                device=device
            )
            evaluation_result.condition_name = condition.condition_name
            
            # Save results
            save_results(
                results_dir=results_dir,
                condition_name=condition.condition_name,
                training_result=training_result,
                evaluation_result=evaluation_result,
                args=args
            )
            
            all_results[condition.condition_name] = evaluation_result
            
            condition_time = time.time() - condition_start
            logger.info(f"Condition {condition.condition_name} completed in {condition_time:.1f}s")
        
        total_time = time.time() - total_start
        logger.info(f"\nAll conditions completed in {total_time:.1f}s")
        
        # Print summary
        print_summary_table(all_results)
        
        # Compute and save comprehensive metrics
        logger.info("Computing comprehensive metrics...")
        all_metrics = compute_all_metrics(all_results)
        
        with open(results_dir / 'comprehensive_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info("Experiment completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    return 0


if __name__ == '__main__':
    exit(main())