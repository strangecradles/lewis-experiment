"""Evaluation metrics for Lewis superadditivity experiment.

Computes accuracy, interaction terms, and all derived metrics.
"""
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

from .models import ModelBank
from .connectors import ComposedSystem
from .config import ConditionConfig
from .utils import get_logger


logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single condition."""
    condition_name: str
    overall_accuracy: float
    accuracy_by_question_type: Dict[str, float]
    total_questions: int
    questions_by_type: Dict[str, int]
    eval_time: float


def evaluate_condition(
    system: ComposedSystem,
    val_loader: DataLoader,
    device: torch.device,
    question_types: Optional[List[str]] = None
) -> EvaluationResult:
    """Evaluate a single condition on validation data.
    
    Args:
        system: Trained composed system
        val_loader: Validation data loader
        device: Evaluation device
        question_types: Question type labels (if available)
        
    Returns:
        EvaluationResult with accuracy metrics
    """
    start_time = time.time()
    logger.info(f"Evaluating condition: {system.__class__.__name__}")
    
    system.eval()
    
    # Track predictions by question type
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    overall_correct = 0
    overall_total = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if len(batch_data) == 3:
                images, questions, answers = batch_data
                batch_question_types = None
            elif len(batch_data) == 4:
                images, questions, answers, batch_question_types = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
                
            images = images.to(device)
            questions = questions.to(device) if questions is not None else None
            answers = answers.to(device)
            
            # Forward pass
            logits = system(images, questions)
            predictions = logits.argmax(dim=1)
            
            # Update overall metrics
            batch_correct = (predictions == answers).cpu().numpy()
            overall_correct += batch_correct.sum()
            overall_total += len(batch_correct)
            
            # Update per-question-type metrics
            if batch_question_types is not None:
                for i, (correct, qtype) in enumerate(zip(batch_correct, batch_question_types)):
                    correct_by_type[qtype] += int(correct)
                    total_by_type[qtype] += 1
            
            if batch_idx % 50 == 0:
                logger.debug(f"Evaluated {batch_idx * val_loader.batch_size} samples")
    
    # Compute accuracies
    overall_accuracy = overall_correct / overall_total
    accuracy_by_type = {}
    
    for qtype, total in total_by_type.items():
        if total > 0:
            accuracy_by_type[qtype] = correct_by_type[qtype] / total
    
    eval_time = time.time() - start_time
    
    logger.info(
        f"Evaluation complete: "
        f"overall_accuracy={overall_accuracy:.4f}, "
        f"total_questions={overall_total}, "
        f"time={eval_time:.1f}s"
    )
    
    if accuracy_by_type:
        logger.info("Accuracy by question type:")
        for qtype, acc in accuracy_by_type.items():
            count = total_by_type[qtype]
            logger.info(f"  {qtype}: {acc:.4f} ({count} questions)")
    
    return EvaluationResult(
        condition_name=getattr(system, 'condition_name', 'unknown'),
        overall_accuracy=overall_accuracy,
        accuracy_by_question_type=accuracy_by_type,
        total_questions=overall_total,
        questions_by_type=dict(total_by_type),
        eval_time=eval_time
    )


def compute_interaction_terms(results_dict: Dict[str, EvaluationResult]) -> Dict[str, float]:
    """Compute pairwise and three-way interaction terms.
    
    Args:
        results_dict: Results keyed by condition name
        
    Returns:
        Dictionary with all interaction terms
    """
    logger.info("Computing interaction terms...")
    
    # Extract accuracies (use overall accuracy as default)
    def get_accuracy(condition_name: str, metric_type: str = 'overall') -> float:
        if condition_name not in results_dict:
            logger.warning(f"Missing results for condition: {condition_name}")
            return 0.0
        
        result = results_dict[condition_name]
        if metric_type == 'overall':
            return result.overall_accuracy
        else:
            return result.accuracy_by_question_type.get(metric_type, 0.0)
    
    # Single model accuracies
    P_A = get_accuracy('A_alone')  # DINOv2
    P_B = get_accuracy('B_alone')  # SigLIP  
    P_C = get_accuracy('C_alone')  # MAE
    
    # Pairwise accuracies
    P_AB = get_accuracy('A+B')
    P_AC = get_accuracy('A+C') 
    P_BC = get_accuracy('B+C')
    
    # Three-way accuracy
    P_ABC = get_accuracy('A+B+C')
    
    # Compute pairwise interaction terms: I₂(X,Y) = P(XY) - P(X) - P(Y)
    I2_AB = P_AB - P_A - P_B
    I2_AC = P_AC - P_A - P_C
    I2_BC = P_BC - P_B - P_C
    
    # Compute three-way interaction term: I₃(A,B,C) = P(ABC) - [P(AB) + P(AC) + P(BC)] + [P(A) + P(B) + P(C)]
    I3_ABC = P_ABC - (P_AB + P_AC + P_BC) + (P_A + P_B + P_C)
    
    interaction_terms = {
        # Individual accuracies
        'P_A': P_A,
        'P_B': P_B, 
        'P_C': P_C,
        # Pairwise accuracies
        'P_AB': P_AB,
        'P_AC': P_AC,
        'P_BC': P_BC,
        # Three-way accuracy
        'P_ABC': P_ABC,
        # Pairwise interactions
        'I2_AB': I2_AB,
        'I2_AC': I2_AC, 
        'I2_BC': I2_BC,
        # Three-way interaction
        'I3_ABC': I3_ABC
    }
    
    logger.info("Interaction terms computed:")
    logger.info(f"  I₃(A,B,C) = {I3_ABC:.4f}")
    logger.info(f"  I₂(A,B) = {I2_AB:.4f}")
    logger.info(f"  I₂(A,C) = {I2_AC:.4f}")
    logger.info(f"  I₂(B,C) = {I2_BC:.4f}")
    
    return interaction_terms


def compute_interaction_terms_by_question_type(
    results_dict: Dict[str, EvaluationResult],
    question_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute interaction terms broken down by question type.
    
    Args:
        results_dict: Results keyed by condition name
        question_types: List of question types to analyze
        
    Returns:
        Dict mapping question_type -> interaction_terms_dict
    """
    logger.info("Computing interaction terms by question type...")
    
    interaction_by_type = {}
    
    for qtype in question_types:
        logger.debug(f"Computing interactions for question type: {qtype}")
        
        # Extract accuracies for this question type
        def get_type_accuracy(condition_name: str) -> float:
            if condition_name not in results_dict:
                return 0.0
            result = results_dict[condition_name]
            return result.accuracy_by_question_type.get(qtype, 0.0)
        
        # Single model accuracies
        P_A = get_type_accuracy('A_alone')
        P_B = get_type_accuracy('B_alone')
        P_C = get_type_accuracy('C_alone')
        
        # Pairwise accuracies  
        P_AB = get_type_accuracy('A+B')
        P_AC = get_type_accuracy('A+C')
        P_BC = get_type_accuracy('B+C')
        
        # Three-way accuracy
        P_ABC = get_type_accuracy('A+B+C')
        
        # Compute interaction terms
        I2_AB = P_AB - P_A - P_B
        I2_AC = P_AC - P_A - P_C
        I2_BC = P_BC - P_B - P_C
        I3_ABC = P_ABC - (P_AB + P_AC + P_BC) + (P_A + P_B + P_C)
        
        interaction_by_type[qtype] = {
            'P_A': P_A, 'P_B': P_B, 'P_C': P_C,
            'P_AB': P_AB, 'P_AC': P_AC, 'P_BC': P_BC,
            'P_ABC': P_ABC,
            'I2_AB': I2_AB, 'I2_AC': I2_AC, 'I2_BC': I2_BC,
            'I3_ABC': I3_ABC
        }
        
        logger.debug(f"  {qtype}: I₃ = {I3_ABC:.4f}")
    
    return interaction_by_type


def compute_all_metrics(results_dict: Dict[str, EvaluationResult]) -> Dict[str, Any]:
    """Compute all derived metrics for the experiment.
    
    Args:
        results_dict: Results for all conditions
        
    Returns:
        Comprehensive metrics dictionary
    """
    logger.info("Computing all derived metrics...")
    
    metrics = {}
    
    # Basic interaction terms
    interaction_terms = compute_interaction_terms(results_dict)
    metrics['interaction_terms'] = interaction_terms
    
    # Get question types from any result
    question_types = []
    for result in results_dict.values():
        if result.accuracy_by_question_type:
            question_types = list(result.accuracy_by_question_type.keys())
            break
    
    if question_types:
        metrics['interaction_by_question_type'] = compute_interaction_terms_by_question_type(
            results_dict, question_types
        )
    
    # Helper function to safely get accuracy
    def safe_get_accuracy(condition_name: str) -> float:
        return results_dict.get(condition_name, EvaluationResult(
            condition_name, 0.0, {}, 0, {}, 0.0
        )).overall_accuracy
    
    # Composition vs. scale comparison
    P_ABC_composed = safe_get_accuracy('A+B+C')
    P_single_large = safe_get_accuracy('single_large') 
    metrics['delta_scale'] = P_ABC_composed - P_single_large
    
    # Composition vs. ensemble comparison
    P_ABC_ensemble = safe_get_accuracy('A+B+C_ensemble')
    metrics['delta_ensemble'] = P_ABC_composed - P_ABC_ensemble
    
    # Connector contribution (with connectors vs concatenation only)
    P_ABC_connectors = P_ABC_composed
    P_ABC_concat = safe_get_accuracy('A+B+C_concat')
    metrics['delta_connector'] = P_ABC_connectors - P_ABC_concat
    
    # Lewis effect (connectors vs concatenation interaction terms)
    if 'A+B+C_concat' in results_dict:
        # Create concatenation results dict for interaction computation
        concat_results = {}
        for condition_name, result in results_dict.items():
            if '_concat' in condition_name:
                base_name = condition_name.replace('_concat', '')
                concat_results[base_name] = result
        
        if len(concat_results) >= 7:  # Need all 7 conditions
            concat_interactions = compute_interaction_terms(concat_results)
            I3_connectors = interaction_terms['I3_ABC']
            I3_concat = concat_interactions['I3_ABC']
            metrics['delta_lewis'] = I3_connectors - I3_concat
        else:
            metrics['delta_lewis'] = None
    else:
        metrics['delta_lewis'] = None
    
    # Diversity effect (diverse models vs copies)
    if 'A+B+C_diverse' in results_dict and 'A+B+C_copies' in results_dict:
        diverse_results = {k: v for k, v in results_dict.items() if '_diverse' in k}
        copies_results = {k: v for k, v in results_dict.items() if '_copies' in k}
        
        if len(diverse_results) >= 7 and len(copies_results) >= 7:
            diverse_interactions = compute_interaction_terms({
                k.replace('_diverse', ''): v for k, v in diverse_results.items()
            })
            copies_interactions = compute_interaction_terms({
                k.replace('_copies', ''): v for k, v in copies_results.items()  
            })
            
            I3_diverse = diverse_interactions['I3_ABC']
            I3_copies = copies_interactions['I3_ABC'] 
            metrics['delta_diversity'] = I3_diverse - I3_copies
        else:
            metrics['delta_diversity'] = None
    else:
        metrics['delta_diversity'] = None
    
    # Log key metrics
    logger.info("Key derived metrics:")
    logger.info(f"  delta_scale = {metrics['delta_scale']:.4f}")
    logger.info(f"  delta_ensemble = {metrics['delta_ensemble']:.4f}")  
    logger.info(f"  delta_connector = {metrics['delta_connector']:.4f}")
    if metrics['delta_lewis'] is not None:
        logger.info(f"  delta_lewis = {metrics['delta_lewis']:.4f}")
    if metrics['delta_diversity'] is not None:
        logger.info(f"  delta_diversity = {metrics['delta_diversity']:.4f}")
    
    return metrics