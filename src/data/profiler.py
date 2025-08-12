# analytics/profiler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json
import time

from core.utils import LoggerFactory, PathManager

@dataclass
class ProfileOptions:
    minimal: bool = True
    title: str = "Dataset profile"
    sample: Optional[int] = None   # e.g., 10000 for speed on huge CSVs
    config_overrides: Optional[Dict[str, Any]] = None  # direct ydata-profiling overrides

class DataProfiler:
    def __init__(self, path_manager: Optional[PathManager] = None):
        self.logger = LoggerFactory.get_logger(__name__)
        self.path_manager = path_manager or PathManager()

    def _make_output_dir(self, base: Optional[str] = None) -> Path:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        out = (Path(base) if base else self.path_manager.artifacts_dir / "analysis") / f"{ts}_profile"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def run(self, df: pd.DataFrame, output_dir: Optional[str] = None, opts: Optional[ProfileOptions] = None) -> Dict[str, str]:
        """
        Run ydata-profiling on df and save HTML + JSON summary into output_dir.
        Returns a dict of file paths.
        """
        try:
            from ydata_profiling import ProfileReport
        except ImportError as e:
            raise ImportError(
                "ydata-profiling is not installed. Install with: pip install ydata-profiling"
            ) from e

        opts = opts or ProfileOptions()
        outdir = self._make_output_dir(output_dir)

        # Optional sampling for speed
        if opts.sample and len(df) > opts.sample:
            self.logger.info(f"Sampling {opts.sample} rows out of {len(df)} for profiling speed.")
            df = df.sample(n=opts.sample, random_state=42).reset_index(drop=True)

        # Reasonable defaults for fast runs when minimal=True
        default_cfg = {
            "dataset": {"title": opts.title},
            "explorative": not opts.minimal,   # minimal=True ==> explorative False
            "samples": {"head": 10, "tail": 10},
            "duplicates": {"enabled": not opts.minimal},
            "interactions": {"continuous": not opts.minimal, "targets": False},
            "correlations": {
                "pearson": {"calculate": True},
                "spearman": {"calculate": not opts.minimal},
                "kendall": {"calculate": False},
                "phi_k": {"calculate": False},
                "cramers": {"calculate": not opts.minimal},
            },
            "missing_diagrams": {"heatmap": not opts.minimal, "dendrogram": False},
        }
        if opts.config_overrides:
            # shallow-merge only top-level keys (keep it simple)
            default_cfg.update(opts.config_overrides)

        self.logger.info("Building profile report...")
        profile = ProfileReport(df, **default_cfg)

        html_path = outdir / "profile.html"
        json_path = outdir / "profile.json"
        meta_path = outdir / "meta.json"
        readme_path = outdir / "README.txt"

        self.logger.info(f"Saving HTML report to {html_path}")
        profile.to_file(str(html_path))

        self.logger.info(f"Saving JSON summary to {json_path}")
        profile.to_file(str(json_path))  # ydata-profiling auto-detects format by extension

        # Small extra: persist a tiny meta file (useful for automation)
        meta = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "minimal": bool(opts.minimal),
            "title": opts.title,
            "output_dir": str(outdir.resolve()),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        with open(readme_path, "w") as f:
            f.write(
                "This directory contains a ydata-profiling analysis.\n"
                "- profile.html : full human-readable report\n"
                "- profile.json : machine-readable summary\n"
                "- meta.json    : small metadata about this run\n"
            )

        return {"html": str(html_path), "json": str(json_path), "meta": str(meta_path), "dir": str(outdir)}
