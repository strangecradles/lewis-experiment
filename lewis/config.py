"""
Configuration for the Lewis superadditivity experiment.

Defines all experimental conditions, model combinations, and connector types
needed to measure three-way interaction terms and controls.
"""

from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum


class ConnectorType(Enum):
    """Types of connectors between models."""
    CROSS_ATTENTION = "cross_attention"
    MLP = "mlp" 
    NONE = "none"  # concatenation only


@dataclass
class ModelSubset:
    """Defines which models are included in a condition."""
    models: Set[str]
    
    def __post_init__(self):
        # Validate model names
        valid_models = {"dino", "siglip", "mae"}
        if not self.models.issubset(valid_models):
            invalid = self.models - valid_models
            raise ValueError(f"Invalid models: {invalid}. Valid: {valid_models}")
    
    @property
    def name(self) -> str:
        """Human-readable name for this model subset."""
        if len(self.models) == 1:
            return list(self.models)[0].upper()
        else:
            sorted_models = sorted(self.models)
            return "".join(m[0].upper() for m in sorted_models)
    
    @property
    def size(self) -> int:
        return len(self.models)


@dataclass
class ConditionConfig:
    """Complete configuration for one experimental condition."""
    condition_name: str
    model_subset: ModelSubset
    connector_type: ConnectorType
    seed: int
    
    # Optional parameters
    num_train_samples: int = None  # None = use all
    num_eval_samples: int = None   # None = use all
    
    def __post_init__(self):
        # Validate connector type makes sense for model subset
        if self.model_subset.size == 1 and self.connector_type != ConnectorType.NONE:
            raise ValueError("Single model conditions must use NONE connector type")


def _generate_main_conditions(seeds: List[int]) -> List[ConditionConfig]:
    """Generate the 7 main combinatorial conditions with cross-attention connectors."""
    model_subsets = [
        # Individual models  
        ModelSubset({"dino"}),
        ModelSubset({"siglip"}), 
        ModelSubset({"mae"}),
        # Pairwise combinations
        ModelSubset({"dino", "siglip"}),
        ModelSubset({"dino", "mae"}),
        ModelSubset({"siglip", "mae"}),
        # Full triad
        ModelSubset({"dino", "siglip", "mae"}),
    ]
    
    conditions = []
    for subset in model_subsets:
        connector_type = ConnectorType.NONE if subset.size == 1 else ConnectorType.CROSS_ATTENTION
        for seed in seeds:
            condition_name = f"main_{subset.name}_seed{seed}"
            conditions.append(ConditionConfig(
                condition_name=condition_name,
                model_subset=subset,
                connector_type=connector_type,
                seed=seed
            ))
    
    return conditions


def _generate_control_conditions(seeds: List[int]) -> List[ConditionConfig]:
    """Generate control conditions."""
    conditions = []
    
    # Control 1: Concatenation baseline (all subsets with no connectors)
    model_subsets = [
        ModelSubset({"dino"}),
        ModelSubset({"siglip"}), 
        ModelSubset({"mae"}),
        ModelSubset({"dino", "siglip"}),
        ModelSubset({"dino", "mae"}),
        ModelSubset({"siglip", "mae"}),
        ModelSubset({"dino", "siglip", "mae"}),
    ]
    
    for subset in model_subsets:
        for seed in seeds:
            condition_name = f"concat_{subset.name}_seed{seed}"
            conditions.append(ConditionConfig(
                condition_name=condition_name,
                model_subset=subset,
                connector_type=ConnectorType.NONE,
                seed=seed
            ))
    
    # Control 3: MLP connectors (parameter-matched to cross-attention)
    pairwise_subsets = [
        ModelSubset({"dino", "siglip"}),
        ModelSubset({"dino", "mae"}), 
        ModelSubset({"siglip", "mae"}),
        ModelSubset({"dino", "siglip", "mae"}),
    ]
    
    for subset in pairwise_subsets:
        for seed in seeds:
            condition_name = f"mlp_{subset.name}_seed{seed}"
            conditions.append(ConditionConfig(
                condition_name=condition_name,
                model_subset=subset,
                connector_type=ConnectorType.MLP,
                seed=seed
            ))
    
    # Control 5: Diversity control (3x DINOv2 copies)
    # Note: This will be handled specially in training - same model loaded 3 times
    # with different initialization seeds
    dino_subsets = [
        ModelSubset({"dino"}),  # Will be treated as dino1
        ModelSubset({"dino"}),  # Will be treated as dino2 
        ModelSubset({"dino"}),  # Will be treated as dino3
        # Pairwise combinations would be dino1+dino2, etc.
        # Full would be dino1+dino2+dino3
    ]
    
    # For simplicity, just add the triad case for diversity control
    for seed in seeds:
        condition_name = f"diversity_DDD_seed{seed}"  # D = DINO
        conditions.append(ConditionConfig(
            condition_name=condition_name,
            model_subset=ModelSubset({"dino", "siglip", "mae"}),  # Will be interpreted as 3x DINO
            connector_type=ConnectorType.CROSS_ATTENTION,
            seed=seed
        ))
    
    return conditions


def _generate_special_conditions(seeds: List[int]) -> List[ConditionConfig]:
    """Generate special baseline conditions."""
    conditions = []
    
    # Control 2: Single large model baseline
    # This will be handled specially - load a larger DINO model
    for seed in seeds:
        condition_name = f"single_large_seed{seed}"
        conditions.append(ConditionConfig(
            condition_name=condition_name,
            model_subset=ModelSubset({"dino"}),  # Will be interpreted as large DINO
            connector_type=ConnectorType.NONE,
            seed=seed
        ))
    
    # Control 4: Ensemble baseline  
    # Train 3 independent model->head mappings, combine logits
    for seed in seeds:
        condition_name = f"ensemble_ABC_seed{seed}"
        conditions.append(ConditionConfig(
            condition_name=condition_name,
            model_subset=ModelSubset({"dino", "siglip", "mae"}),
            connector_type=ConnectorType.NONE,  # Will be handled specially
            seed=seed
        ))
    
    return conditions


def get_all_conditions(seeds: List[int] = [42, 123, 456]) -> List[ConditionConfig]:
    """Generate all experimental conditions.
    
    Args:
        seeds: Random seeds to use for each condition
        
    Returns:
        List of all condition configurations
    """
    conditions = []
    conditions.extend(_generate_main_conditions(seeds))
    conditions.extend(_generate_control_conditions(seeds))  
    conditions.extend(_generate_special_conditions(seeds))
    
    return conditions


def get_condition_groups() -> Dict[str, List[str]]:
    """Group conditions by experimental purpose for analysis.
    
    Returns:
        Dictionary mapping group names to condition name patterns
    """
    return {
        # Main combinatorial experiment (for computing I3)
        "main_single": ["main_DINO_seed*", "main_SIGLIP_seed*", "main_MAE_seed*"],
        "main_pairwise": ["main_DS_seed*", "main_DM_seed*", "main_MS_seed*"],
        "main_triad": ["main_DMS_seed*"],
        
        # Concatenation controls (for computing concatenation I3)
        "concat_single": ["concat_DINO_seed*", "concat_SIGLIP_seed*", "concat_MAE_seed*"],
        "concat_pairwise": ["concat_DS_seed*", "concat_DM_seed*", "concat_MS_seed*"],
        "concat_triad": ["concat_DMS_seed*"],
        
        # MLP controls
        "mlp_pairwise": ["mlp_DS_seed*", "mlp_DM_seed*", "mlp_MS_seed*"],
        "mlp_triad": ["mlp_DMS_seed*"],
        
        # Special controls
        "single_large": ["single_large_seed*"],
        "ensemble": ["ensemble_ABC_seed*"],
        "diversity": ["diversity_DDD_seed*"],
    }


def get_interaction_subsets() -> Dict[str, List[str]]:
    """Get model subsets needed for computing interaction terms.
    
    Returns:
        Dictionary mapping interaction terms to required model subset names
    """
    return {
        # For computing I3(A,B,C) = P(ABC) - [P(AB) + P(AC) + P(BC)] + [P(A) + P(B) + P(C)]
        "three_way": {
            "P_ABC": "DMS",
            "P_AB": "DS", 
            "P_AC": "DM",
            "P_BC": "MS",
            "P_A": "DINO",
            "P_B": "SIGLIP", 
            "P_C": "MAE"
        },
        
        # For computing pairwise I2 terms
        "pairwise_AB": {
            "P_AB": "DS",
            "P_A": "DINO",
            "P_B": "SIGLIP"
        },
        "pairwise_AC": {
            "P_AC": "DM", 
            "P_A": "DINO",
            "P_C": "MAE"
        },
        "pairwise_BC": {
            "P_BC": "MS",
            "P_B": "SIGLIP",
            "P_C": "MAE"
        }
    }


def get_model_mapping() -> Dict[str, str]:
    """Map short model names to full model identifiers."""
    return {
        "dino": "facebook/dinov2-vit-small-patch14-336",
        "siglip": "google/siglip-vit-base-patch16-224", 
        "mae": "facebook/vit-mae-base"
    }


def print_experiment_overview():
    """Print summary of experimental design."""
    seeds = [42, 123, 456]
    conditions = get_all_conditions(seeds)
    
    print("Lewis Superadditivity Experiment Overview")
    print("=" * 50)
    print(f"Total conditions: {len(conditions)}")
    print(f"Seeds per condition: {len(seeds)}")
    
    groups = get_condition_groups()
    for group_name, patterns in groups.items():
        matching_conditions = []
        for c in conditions:
            for pattern in patterns:
                pattern_base = pattern.replace('_seed*', '').replace('*', '')
                if c.condition_name.startswith(pattern_base):
                    matching_conditions.append(c)
                    break
        print(f"{group_name}: {len(matching_conditions)} conditions")
    
    print("\nModel subsets:")
    unique_subsets = set(c.model_subset.name for c in conditions)
    for subset in sorted(unique_subsets):
        count = sum(1 for c in conditions if c.model_subset.name == subset)
        print(f"  {subset}: {count} conditions")
    
    print(f"\nConnector types:")
    for conn_type in ConnectorType:
        count = sum(1 for c in conditions if c.connector_type == conn_type)
        print(f"  {conn_type.value}: {count} conditions")


if __name__ == "__main__":
    print_experiment_overview()