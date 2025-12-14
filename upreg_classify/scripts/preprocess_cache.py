"""Build and cache the full preprocessed dataframe once.

This decouples expensive preprocessing from training runs.

Example:
	python upreg_classify\\scripts\\preprocess_cache.py --area NO1 --out upreg_classify\\data\\preprocessed\\df_NO1.pkl

Then train using:
	python upreg_classify\\src\\train\\train.py --task multiclass --preprocessed_path upreg_classify\\data\\preprocessed\\df_NO1.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Import of project modules regardless of current working dir
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)

from src.data.preprocess import Config, preprocess_all


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Precompute and cache the preprocessed dataframe")
	p.add_argument("--area", type=str, default="NO1", choices=["NO1", "NO2"])
	p.add_argument(
		"--data_dir",
		type=str,
		default=str(Path("upreg_classify") / "data" / "raw"),
		help="Raw data directory used by preprocessing",
	)
	p.add_argument("--include_2024", action="store_true")
	p.add_argument("--no-include_2024", dest="include_2024", action="store_false")
	p.set_defaults(include_2024=True)
	p.add_argument("--heavy_interactions", action="store_true")
	p.add_argument("--dropna", action="store_true")
	p.set_defaults(dropna=True)
	p.add_argument("--activation_lag_start", type=int, default=4)
	p.add_argument("--single_persistence", action="store_true")
	p.add_argument("--min_up_volume", type=float, default=None)
	p.add_argument("--min_down_volume", type=float, default=None)
	p.add_argument(
		"--out",
		type=str,
		required=True,
		help="Output cache path (pickle), e.g. upreg_classify\\data\\preprocessed\\df_NO1.pkl",
	)
	p.add_argument("--force", action="store_true", help="Overwrite if cache already exists")
	return p.parse_args()


def main() -> None:
	args = parse_args()

	out_path = Path(args.out)
	if out_path.exists() and not args.force:
		raise SystemExit(f"Refusing to overwrite existing cache without --force: {out_path}")

	cfg = Config(
		data_dir=args.data_dir,
		area=args.area,
		include_2024=args.include_2024,
		heavy_interactions=args.heavy_interactions,
		dropna=args.dropna,
		activation_lag_start=args.activation_lag_start,
		single_persistence=args.single_persistence,
		min_up_volume=args.min_up_volume,
		min_down_volume=args.min_down_volume,
	)

	df = preprocess_all(cfg, cache_path=str(out_path), force_recompute=True)
	print(f"Cached dataframe: {out_path} (rows={len(df)}, cols={df.shape[1]})")


if __name__ == "__main__":
	main()