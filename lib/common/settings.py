from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Settings:
    paths: dict
    betting: dict
    decisions: dict
    lgbm: dict | None = None
    features: dict | None = None

    @classmethod
    def load(cls, path: str | Path = "config/default.yaml") -> "Settings":
        path = Path(path)
        cfg = yaml.safe_load(path.read_text()) if path.exists() else {}

        # Prefer Week-2 style paths: { vendors, warehouse, artifacts, mlruns, reports }
        paths = dict(cfg.get("paths", {}))

        # Map any Week-1 legacy keys into the paths dict if present
        legacy_map = {
            "raw": cfg.get("raw_dir"),
            "warehouse": cfg.get("warehouse_dir"),
            "artifacts": cfg.get("artifacts_dir"),
            "mlruns": cfg.get("mlruns_dir"),
            "reports": cfg.get("reports_dir"),
        }
        for k, v in legacy_map.items():
            if v and k not in paths:
                paths[k] = v

        # Defaults for missing entries
        defaults = {
            "vendors": "data/vendors",
            "raw": "data/raw",
            "warehouse": "data/warehouse",
            "artifacts": "artifacts",
            "mlruns": "mlruns",
            "reports": "reports",
        }
        for k, v in defaults.items():
            paths.setdefault(k, v)

        betting = cfg.get("betting", {})
        decisions = cfg.get("decisions", {"pregame_offset_min": 30})
        lgbm = cfg.get("lgbm", {})                # ✅ new
        features = cfg.get("features", {})        # ✅ optional new

        return Settings(
            paths=paths,
            betting=betting,
            decisions=decisions,
            lgbm=lgbm,
            features=features
        )

    # ------------ Back-compat properties (Week-1 code expects these) ------------
    @property
    def raw_dir(self) -> str:
        return self.paths.get("raw", "data/raw")

    @property
    def warehouse_dir(self) -> str:
        return self.paths["warehouse"]

    @property
    def artifacts_dir(self) -> str:
        return self.paths["artifacts"]

    @property
    def mlruns_dir(self) -> str:
        return self.paths["mlruns"]

    @property
    def reports_dir(self) -> str:
        return self.paths["reports"]

    # Some Week-1 modules referenced a single features file
    @property
    def warehouse_features(self) -> str:
        return str(Path(self.warehouse_dir) / "features.parquet")


# Convenience wrapper used by newer modules
def load_settings() -> Settings:
    return Settings.load()
