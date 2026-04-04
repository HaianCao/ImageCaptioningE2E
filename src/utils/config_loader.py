"""
Configuration loading utilities using OmegaConf.

Supports loading YAML configs and merging with command-line overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of config overrides

    Returns:
        OmegaConf DictConfig object

    Example:
        config = load_config("configs/config.yaml", {"device": "cpu"})
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    config = OmegaConf.load(config_path)

    # Apply overrides if provided
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    # Resolve environment variables in paths
    config = _resolve_env_vars(config)

    return config


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple OmegaConf configs.

    Args:
        *configs: Variable number of DictConfig objects

    Returns:
        Merged DictConfig
    """
    if not configs:
        return OmegaConf.create()

    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


def load_task_configs(
    base_config_path: str = "configs/config.yaml",
    task_config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load base config and optionally merge with task-specific config.

    Args:
        base_config_path: Path to base config.yaml
        task_config_path: Path to task-specific config (optional)
        overrides: Config overrides

    Returns:
        Merged config
    """
    base_config = load_config(base_config_path, overrides)

    if task_config_path:
        task_config = load_config(task_config_path)
        return merge_configs(base_config, task_config)

    return base_config


def _resolve_env_vars(config: DictConfig) -> DictConfig:
    """
    Resolve environment variables in string values.

    Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
    """
    def resolve_value(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        else:
            return value

    resolved = OmegaConf.create(resolve_value(OmegaConf.to_container(config)))
    return resolved


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save config to YAML file.

    Args:
        config: OmegaConf config to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, output_path)
    print(f"Config saved to: {output_path}")


def print_config(config: DictConfig, resolve: bool = True) -> None:
    """
    Pretty print configuration.

    Args:
        config: Config to print
        resolve: Whether to resolve interpolations
    """
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(OmegaConf.to_yaml(config, resolve=resolve))
    print("=" * 50)