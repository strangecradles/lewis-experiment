# Code Style
- Python 3.11+, type hints on all function signatures
- Use dataclasses for configuration, not dicts
- Keep each file under 300 lines
- Use torch.no_grad() for all frozen model forward passes
- Print progress to stdout (no logging framework needed)
- Use pathlib.Path for all file paths
