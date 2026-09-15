import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(cfg: dict) -> torch.dtype:
    if cfg["training"].get("bf16", False) and torch.cuda.is_available():
        return torch.bfloat16
    if cfg["training"].get("fp16", False) and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def ensure_output_dirs(cfg: dict) -> dict:
    base = Path(cfg["training"]["output_dir"])
    dirs = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "adapter": base / "adapter",
        "metrics": base / "metrics",
        "logs": base / "logs",
        "figures": base / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
