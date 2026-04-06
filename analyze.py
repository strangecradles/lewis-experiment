#!/usr/bin/env python3
"""Post-hoc analysis of Lewis superadditivity experiment results.

Loads results, computes interaction terms, generates plots and statistical tests.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from lewis.evaluate import compute_interaction_terms, compute_interaction_terms_by_question_type
from lewis.utils import get_logger


# Set matplotlib backend for headless environments
plt.switch_backend('Agg')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze results from Lewis superadditivity experiment"
    )
    
    parser.add_argument(
        'results_dir',
        type=str,
        help='Directory containing experiment results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis (default: results_dir/analysis)'
    )
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def load_all_results(results_dir: Path) -> Dict[str, Any]:
    """Load all experimental results from directory.
    
    Args:
        results_dir: Directory containing condition subdirectories
        
    Returns:
        Dictionary with condition_name -> evaluation_results
    """
    logger = get_logger(__name__)
    logger.info(f"Loading results from: {results_dir}")
    
    results = {}
    
    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
            continue
            
        eval_file = condition_dir / 'evaluation.json'
        if not eval_file.exists():
            logger.warning(f"No evaluation.json found in {condition_dir}")
            continue
        
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            # Create a simple result object
            from lewis.evaluate import EvaluationResult
            result = EvaluationResult(
                condition_name=eval_data['condition_name'],
                overall_accuracy=eval_data['overall_accuracy'],
                accuracy_by_question_type=eval_data['accuracy_by_question_type'],
                total_questions=eval_data['total_questions'],
                questions_by_type=eval_data['questions_by_type'],
                eval_time=eval_data['eval_time']
            )
            
            results[condition_dir.name] = result
            logger.debug(f"Loaded {condition_dir.name}: {result.overall_accuracy:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to load {condition_dir}: {e}")
    
    logger.info(f"Loaded {len(results)} conditions")
    return results


def print_interaction_tables(
    interaction_terms: Dict[str, float],
    interaction_by_type: Optional[Dict[str, Dict[str, float]]] = None
):
    """Print formatted tables of interaction terms."""
    logger = get_logger(__name__)
    
    # Overall interaction table
    logger.info("\n" + "="*60)
    logger.info("INTERACTION TERMS (Overall)")
    logger.info("="*60)
    
    # Individual model performance
    logger.info("Individual Model Performance:")
    logger.info(f"  P(A) [DINOv2]:    {interaction_terms['P_A']:.4f}")
    logger.info(f"  P(B) [SigLIP]:    {interaction_terms['P_B']:.4f}")
    logger.info(f"  P(C) [MAE]:       {interaction_terms['P_C']:.4f}")
    
    # Pairwise performance
    logger.info("\nPairwise Composition Performance:")
    logger.info(f"  P(A+B):           {interaction_terms['P_AB']:.4f}")
    logger.info(f"  P(A+C):           {interaction_terms['P_AC']:.4f}")
    logger.info(f"  P(B+C):           {interaction_terms['P_BC']:.4f}")
    
    # Three-way performance
    logger.info("\nThree-way Composition Performance:")
    logger.info(f"  P(A+B+C):         {interaction_terms['P_ABC']:.4f}")
    
    # Interaction terms
    logger.info("\nInteraction Terms:")
    logger.info(f"  I₂(A,B):          {interaction_terms['I2_AB']:+.4f}")
    logger.info(f"  I₂(A,C):          {interaction_terms['I2_AC']:+.4f}")
    logger.info(f"  I₂(B,C):          {interaction_terms['I2_BC']:+.4f}")
    logger.info(f"  I₃(A,B,C):        {interaction_terms['I3_ABC']:+.4f}")
    
    # Interpretation
    I3 = interaction_terms['I3_ABC']
    if I3 > 0.01:
        interpretation = "STRONG superadditive (Lewis effect detected!)"
    elif I3 > 0.001:
        interpretation = "WEAK superadditive (marginal Lewis effect)"
    elif I3 > -0.001:
        interpretation = "ADDITIVE (no Lewis effect)"
    else:
        interpretation = "SUBADDITIVE (negative interaction)"
    
    logger.info(f"\nInterpretation: {interpretation}")
    
    # Question type breakdown
    if interaction_by_type:
        logger.info("\n" + "="*60)
        logger.info("INTERACTION TERMS BY QUESTION TYPE")
        logger.info("="*60)
        
        # Sort by I3 value for easier reading
        sorted_types = sorted(
            interaction_by_type.items(),
            key=lambda x: x[1]['I3_ABC'],
            reverse=True
        )
        
        header = f"{'Question Type':<20} {'I₃':<8} {'I₂(A,B)':<8} {'I₂(A,C)':<8} {'I₂(B,C)':<8} {'P(ABC)':<8}"
        logger.info(header)
        logger.info("-" * len(header))
        
        for qtype, terms in sorted_types:
            row = f"{qtype:<20} {terms['I3_ABC']:+.4f} {terms['I2_AB']:+.4f} {terms['I2_AC']:+.4f} {terms['I2_BC']:+.4f} {terms['P_ABC']:.4f}"
            logger.info(row)
        
        # Highlight strongest superadditive question types
        strong_superadditive = [
            (qtype, terms['I3_ABC']) 
            for qtype, terms in interaction_by_type.items()
            if terms['I3_ABC'] > 0.01
        ]
        
        if strong_superadditive:
            logger.info(f"\nStrongest superadditive question types:")
            for qtype, i3 in sorted(strong_superadditive, key=lambda x: x[1], reverse=True):
                logger.info(f"  {qtype}: I₃ = {i3:+.4f}")


def statistical_analysis(
    interaction_terms: Dict[str, float],
    interaction_by_type: Optional[Dict[str, Dict[str, float]]] = None,
    alpha: float = 0.05
):
    """Perform statistical analysis of interaction terms."""
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*60)
    
    # Test overall I₃ > 0
    I3_overall = interaction_terms['I3_ABC']
    
    # For a single run, we can't do a proper t-test
    # But we can provide interpretation
    logger.info("Overall Three-way Interaction Test:")
    logger.info(f"  H₀: I₃ ≤ 0 (no superadditivity)")
    logger.info(f"  H₁: I₃ > 0 (superadditivity)")
    logger.info(f"  Observed I₃: {I3_overall:+.6f}")
    
    if I3_overall > 0:
        logger.info(f"  Result: Evidence FOR superadditivity")
    else:
        logger.info(f"  Result: No evidence for superadditivity")
    
    logger.info(f"  Note: Proper statistical testing requires multiple seeds/runs")
    
    # Question type analysis
    if interaction_by_type:
        logger.info("\nQuestion Type Analysis:")
        
        # Count positive vs negative I₃ values
        positive_i3 = sum(1 for terms in interaction_by_type.values() if terms['I3_ABC'] > 0)
        total_types = len(interaction_by_type)
        
        logger.info(f"  Question types with I₃ > 0: {positive_i3}/{total_types} ({positive_i3/total_types:.1%})")
        
        # Effect sizes
        i3_values = [terms['I3_ABC'] for terms in interaction_by_type.values()]
        logger.info(f"  Mean I₃ across question types: {np.mean(i3_values):+.4f}")
        logger.info(f"  Std I₃ across question types: {np.std(i3_values):.4f}")
        logger.info(f"  Max I₃: {np.max(i3_values):+.4f}")
        logger.info(f"  Min I₃: {np.min(i3_values):+.4f}")


def create_plots(
    interaction_terms: Dict[str, float],
    interaction_by_type: Optional[Dict[str, Dict[str, float]]] = None,
    output_dir: Path = Path('.')
):
    """Generate analysis plots."""
    logger = get_logger(__name__)
    logger.info("Generating plots...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Plot 1: I₃ by question type (bar chart)
    if interaction_by_type:
        plt.figure(figsize=(12, 8))
        
        qtypes = list(interaction_by_type.keys())
        i3_values = [interaction_by_type[qt]['I3_ABC'] for qt in qtypes]
        
        # Sort by I₃ value
        sorted_data = sorted(zip(qtypes, i3_values), key=lambda x: x[1], reverse=True)
        qtypes_sorted, i3_sorted = zip(*sorted_data)
        
        # Color bars by sign
        colors = ['green' if i3 > 0 else 'red' for i3 in i3_sorted]
        
        bars = plt.bar(range(len(qtypes_sorted)), i3_sorted, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.xlabel('Question Type')
        plt.ylabel('Three-way Interaction Term (I₃)')
        plt.title('Superadditivity by Question Type\n(Green = Superadditive, Red = Subadditive)')
        plt.xticks(range(len(qtypes_sorted)), qtypes_sorted, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'i3_by_question_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: All interaction terms by question type (heatmap)
        plt.figure(figsize=(14, 10))
        
        # Prepare data matrix
        interaction_types = ['I2_AB', 'I2_AC', 'I2_BC', 'I3_ABC']
        interaction_labels = ['I₂(A,B)', 'I₂(A,C)', 'I₂(B,C)', 'I₃(A,B,C)']
        
        data_matrix = []
        for qt in qtypes_sorted:
            row = [interaction_by_type[qt][it] for it in interaction_types]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        sns.heatmap(
            data_matrix,
            xticklabels=interaction_labels,
            yticklabels=qtypes_sorted,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Interaction Term Value'}
        )
        
        plt.title('All Interaction Terms by Question Type')
        plt.xlabel('Interaction Term')
        plt.ylabel('Question Type')
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'interaction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Composition hierarchy
    plt.figure(figsize=(10, 8))
    
    # Performance levels
    individual_perf = [
        interaction_terms['P_A'],
        interaction_terms['P_B'], 
        interaction_terms['P_C']
    ]
    
    pairwise_perf = [
        interaction_terms['P_AB'],
        interaction_terms['P_AC'],
        interaction_terms['P_BC']
    ]
    
    three_way_perf = [interaction_terms['P_ABC']]
    
    # Box plot style visualization
    positions = [1, 2, 3]
    box_data = [individual_perf, pairwise_perf, three_way_perf]
    labels = ['Individual\n(A, B, C)', 'Pairwise\n(AB, AC, BC)', 'Three-way\n(ABC)']
    
    bp = plt.boxplot(box_data, positions=positions, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Accuracy')
    plt.title('Performance by Composition Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(figures_dir / 'composition_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to: {figures_dir}")


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Setup logging
    from lewis.utils import setup_logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    logger = get_logger(__name__)
    
    logger.info("Starting Lewis experiment analysis")
    
    # Setup directories
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load results
        all_results = load_all_results(results_dir)
        if not all_results:
            logger.error("No results found")
            return 1
        
        # Compute interaction terms
        logger.info("Computing interaction terms...")
        interaction_terms = compute_interaction_terms(all_results)
        
        # Get question types
        question_types = set()
        for result in all_results.values():
            question_types.update(result.accuracy_by_question_type.keys())
        question_types = sorted(list(question_types))
        
        interaction_by_type = None
        if question_types:
            interaction_by_type = compute_interaction_terms_by_question_type(
                all_results, question_types
            )
        
        # Print analysis tables
        print_interaction_tables(interaction_terms, interaction_by_type)
        
        # Statistical analysis
        statistical_analysis(interaction_terms, interaction_by_type, args.alpha)
        
        # Generate plots
        if not args.skip_plots:
            create_plots(interaction_terms, interaction_by_type, output_dir)
        
        # Save analysis results
        analysis_results = {
            'interaction_terms': interaction_terms,
            'interaction_by_question_type': interaction_by_type,
            'summary': {
                'overall_i3': interaction_terms['I3_ABC'],
                'superadditive': interaction_terms['I3_ABC'] > 0,
                'effect_magnitude': 'strong' if interaction_terms['I3_ABC'] > 0.01 else 'weak',
                'num_conditions': len(all_results),
                'num_question_types': len(question_types)
            }
        }
        
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Analysis completed. Results saved to: {output_dir}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        
        I3 = interaction_terms['I3_ABC']
        logger.info(f"Overall three-way interaction (I₃): {I3:+.6f}")
        
        if I3 > 0.01:
            logger.info("CONCLUSION: Strong evidence for Lewis superadditivity!")
            logger.info("The composed system exhibits emergent capabilities.")
        elif I3 > 0.001:
            logger.info("CONCLUSION: Weak evidence for Lewis superadditivity.")
            logger.info("Marginal emergent effects detected.")
        elif I3 > -0.001:
            logger.info("CONCLUSION: No evidence for superadditivity.")
            logger.info("Performance is merely additive.")
        else:
            logger.info("CONCLUSION: Negative interaction detected.")
            logger.info("Composition interferes with individual capabilities.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())