"""Shared helpers for notebook-driven Visual Genome pipelines."""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch


def resolve_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the repository root that contains both configs/ and src/."""
    current = (start_path or Path.cwd()).resolve()

    while current != current.parent:
        if (current / "configs").exists() and (current / "src").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Không tìm thấy thư mục gốc dự án chứa configs/ và src/.")


def configure_notebook_environment(
    start_path: Optional[Path] = None,
    *,
    clear_src_modules: bool = False,
) -> Path:
    """Add the project root to sys.path and optionally clear cached src imports."""
    project_root = resolve_project_root(start_path)
    project_root_text = str(project_root)

    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    os.chdir(project_root_text)

    if clear_src_modules:
        for module_name in [name for name in list(sys.modules) if name == "src" or name.startswith("src.")]:
            del sys.modules[module_name]
        importlib.invalidate_caches()

    return project_root


def get_device(preferred_device: str = "cuda") -> str:
    """Return a safe execution device with CPU fallback."""
    preferred_device = str(preferred_device).lower().strip()
    if preferred_device.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs for notebook reruns."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_input_mode(mode: str) -> str:
    """Normalize the notebook input-mode aliases to the canonical set."""
    mode = str(mode).lower().strip()
    aliases = {
        "grayscale": "gray",
        "grey": "gray",
        "edge": "contour",
        "edges": "contour",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"rgb", "gray", "contour"}:
        raise ValueError("input mode phải là 'rgb', 'gray', hoặc 'contour'")
    return mode


def require_files(paths: Sequence[Path], label: str) -> None:
    """Raise if any file in paths does not exist."""
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu {label}: {missing}")


def ensure_nonempty_cache(cache_path: Path) -> bool:
    """Return True when a cache file exists and contains at least one entry."""
    cache_path = Path(cache_path)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return False

    try:
        cache = torch.load(cache_path, map_location="cpu")
        return bool(cache)
    except Exception:
        return False


def collect_split_image_ids(processed_root: Path, split_names: Sequence[str]) -> List[int]:
    """Collect unique image ids referenced by the requested split annotation files."""
    image_ids = set()

    for split_name in split_names:
        annotation_file = Path(processed_root) / split_name / "annotations.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Thiếu annotation: {annotation_file}")

        with open(annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = raw if isinstance(raw, list) else raw.get("samples", [])
        image_ids.update(sample["image_id"] for sample in samples)

    return sorted(image_ids)


def missing_local_image_ids(image_dir: Path, image_ids: Iterable[int]) -> List[int]:
    """Return the ids for images that are not present locally."""
    image_dir = Path(image_dir)
    return [image_id for image_id in image_ids if not (image_dir / f"{image_id}.jpg").exists()]
