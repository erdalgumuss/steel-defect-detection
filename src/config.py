# src/config.py
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, yaml_path: str = "config.yaml"):
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        # normalize numeric fields (örneğin "1e-4" string olursa float'a çevir)
        cfg = self._normalize(cfg)
        self._cfg = cfg

    def __getitem__(self, key: str):
        return self._cfg[key]

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    @property
    def dict(self) -> Dict[str, Any]:
        return self._cfg

    def _normalize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize YAML-loaded config (str -> float if numeric)."""
        for k, v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = self._normalize(v)
            elif isinstance(v, str):
                # try to cast scientific notation / numeric strings
                try:
                    if "." in v or "e" in v.lower():
                        cfg[k] = float(v)
                    else:
                        cfg[k] = int(v)
                except ValueError:
                    pass  # keep as string
        return cfg
