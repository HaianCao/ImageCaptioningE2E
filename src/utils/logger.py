"""
Logging utilities for experiment tracking.

Supports TensorBoard and Weights & Biases.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Unified logging interface for experiments.

    Supports console logging, TensorBoard, and Weights & Biases.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "vg_caption",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "visual-genome-caption",
        config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup console logging
        self._setup_console_logging()

        # Setup TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            tb_dir = self.log_dir / "tensorboard" / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            self.info(f"TensorBoard logging to: {tb_dir}")

        # Setup Weights & Biases
        self.wandb_run = None
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                self.warning("Weights & Biases not available. Install with: pip install wandb")
            else:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config,
                    dir=str(self.log_dir)
                )
                self.info(f"W&B logging to project: {wandb_project}")

    def _setup_console_logging(self):
        """Setup console logging with proper formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(self.experiment_name)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        Log metrics to all enabled loggers.

        Args:
            metrics: Dict of metric names to values
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # Weights & Biases
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

        # Console
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items())
        if prefix == "epoch":
            self.info(f"Epoch {step + 1}: {metrics_str}")
        else:
            self.info(f"Step {step}: {metrics_str}")

    def log_image(self, tag: str, image, step: int):
        """
        Log image to TensorBoard.

        Args:
            tag: Image tag
            image: PIL Image or tensor
            step: Current step
        """
        if self.tb_writer:
            self.tb_writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values, step: int):
        """
        Log histogram to TensorBoard.

        Args:
            tag: Histogram tag
            values: Values to plot
            step: Current step
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_logger(
    log_dir: str = "logs",
    experiment_name: str = "vg_caption",
    **kwargs
) -> Logger:
    """
    Convenience function to create a logger.

    Args:
        log_dir: Directory for logs
        experiment_name: Name of experiment
        **kwargs: Additional arguments for Logger

    Returns:
        Logger instance
    """
    return Logger(log_dir=log_dir, experiment_name=experiment_name, **kwargs)