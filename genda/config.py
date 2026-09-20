"""YAML configuration loading shared by training and inference."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "config_GenDA.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = ("hidden_dim", "layers", "heads", "max_token_len", "kmer_size")
    for key in required:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["hidden_dim"] % config["heads"]:
        raise ValueError("hidden_dim must be divisible by heads")
    if config.get("backbone", "modernbert") not in ("modernbert", "llama"):
        raise ValueError("backbone must be modernbert or llama")
    return config


def checkpoint_config(checkpoint, explicit=None):
    path = Path(explicit) if explicit else Path(checkpoint).parent / "run_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint configuration missing: {path}; provide --config")
    return load_config(path)
