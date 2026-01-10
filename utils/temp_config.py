import os
import tempfile
import random
import math
import yaml
from contextlib import contextmanager
from typing import Optional
import datetime

__all__ = ["generate_temp_config", "temp_config_file"]

def _pick_behavior(cfg: dict) -> str:
    behaviors = cfg.get("behaviors", {})
    if not behaviors:
        raise RuntimeError("No behaviors are found in config to randomize.")
    return next(iter(behaviors))

def _rand_log_uniform(low: float, high: float) -> float:
    return 10 ** random.uniform(math.log10(low), math.log10(high))

def _randomize_behavior(bcfg: dict) -> None:
    hp = bcfg.setdefault("hyperparameters", {})
    hp["batch_size"] = int(random.choice([32, 64, 120, 128, 256]))
    hp["buffer_size"] = int(max(hp["batch_size"] * random.choice([8, 10, 20]), random.choice([1000, 5000, 10000])))
    hp["learning_rate"] = float(f"{_rand_log_uniform(1e-5, 3e-3):.6f}")
    hp["beta"] = float(f"{random.uniform(1e-4, 1e-2):.6f}")
    hp["epsilon"] = float(f"{random.uniform(0.08, 0.3):.3f}")
    hp.setdefault("num_epoch", 3)

    net = bcfg.setdefault("network_settings", {})
    net["hidden_units"] = int(random.choice([64, 128, 256]))
    net["num_layers"] = int(random.choice([1, 2, 3]))

    if "max_steps" in bcfg:
        base = int(bcfg.get("max_steps", 5_00_000))
        delta = int(base * random.uniform(-0.5, 0.5))
        bcfg["max_steps"] = max(10_000, min(2_000_000, base + delta))

def generate_temp_config(base_config_path: str, run_id: Optional[str] = None, seed: Optional[int] = None) -> str:
    """
    Read base YAML, apply small randomized overrides and write to system temp dir.
    Returns path to the temp YAML file.
    """
    if seed is not None:
        random.seed(seed)

    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    behavior_name = _pick_behavior(cfg)
    _randomize_behavior(cfg["behaviors"][behavior_name])

    if run_id is None:
        run_id = datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")

    tmp_dir = tempfile.gettempdir()
    temp_path = os.path.join(tmp_dir, f"{run_id}_config.yaml")
    with open(temp_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return temp_path

@contextmanager
def temp_config_file(base_config_path: str, run_id: Optional[str] = None, seed: Optional[int] = None):
    """
    Context manager: yields a temp config path and removes it on exit unless KEEP_TEMP_CONFIG env var set.
    """
    path = generate_temp_config(base_config_path, run_id=run_id, seed=seed)
    try:
        yield path
    finally:
        if os.getenv("KEEP_TEMP_CONFIG"):
            return
        try:
            os.remove(path)
        except OSError:
            pass
