"""Regenerate the few-shot table and vector/raster figure from frozen results.

Run from any directory. This reads numerical summaries only and performs no
episode sampling, fitting, checkpoint selection or held-out score tuning.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.harp_fewshot import paper_outputs


if __name__ == "__main__":
    report = json.loads((ROOT / "results/harp_revision/v2/fewshot/report.json").read_text(encoding="utf-8"))
    cfg = report["episode_config"]
    paper_outputs(report["summary"], cfg["shots"], ROOT / "paper/manuscript_harp",
                  nway=cfg["nway"], queries=cfg["queries"], episodes=cfg["episodes"])
