"""
Utility functions for the Lewis superadditivity experiment.

Provides seeding, device detection, result I/O, and timing utilities.
"""

import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

import torch
import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    """Alias for set_seed for compatibility."""
    set_seed(seed)


def get_device_info() -> dict:
    """Return device info dict."""
    device = get_device()
    info = {"device": str(device), "type": device.type}
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
    return info


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_results(condition: str, metrics: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save experimental results to JSON file.
    
    Args:
        condition: Name/identifier of the experimental condition
        metrics: Dictionary of metrics to save
        path: Output file path (will create parent directories)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results or create new
    if path.exists():
        with open(path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Add this condition's results
    results[condition] = {
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save updated results
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serialize)


def load_results(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load all experimental results from JSON file.
    
    Returns:
        Dictionary mapping condition names to their results
    """
    path = Path(path)
    if not path.exists():
        return {}
    
    with open(path, 'r') as f:
        return json.load(f)


def _json_serialize(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@contextmanager
def Timer(name: Optional[str] = None):
    """Context manager for timing code blocks.
    
    Usage:
        with Timer("training"):
            # code to time
            pass
    
    Or get the time:
        with Timer("training") as timer:
            # code to time
            pass
        print(f"Took {timer.elapsed:.2f} seconds")
    """
    start_time = time.time()
    timer_obj = _TimerResult()
    
    try:
        yield timer_obj
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        timer_obj.elapsed = elapsed
        
        if name:
            print(f"{name}: {elapsed:.2f}s")


class _TimerResult:
    """Helper class to store timer results."""
    def __init__(self):
        self.elapsed = 0.0


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with appropriate suffixes."""
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.{precision}f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    else:
        return f"{num/1_000_000_000:.{precision}f}B"


def print_model_info(model: torch.nn.Module, name: str) -> None:
    """Print model parameter count and memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{name}:")
    print(f"  Total parameters: {format_number(total_params)}")
    print(f"  Trainable parameters: {format_number(trainable_params)}")
    print(f"  Memory (approx): {total_params * 4 / (1024**2):.1f} MB")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())