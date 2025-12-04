from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# Fixed class order for ternary tasks used across notebooks and scripts
CLASSES_DEFAULT: List[str] = ["up", "down", "none"]


def resolve_models_dir(preferred: Path, fallbacks: Sequence[Path] | None = None) -> Path:
	"""Return the first existing models directory among preferred and fallbacks.

	Raises FileNotFoundError if none exist.
	"""
	if preferred and Path(preferred).exists():
		return Path(preferred)
	if fallbacks:
		for p in fallbacks:
			p = Path(p)
			if p.exists():
				return p
	raise FileNotFoundError("No valid models directory found among preferred and fallbacks")


def load_predictions_csv(models_dir: Path, split: str) -> pd.DataFrame:
	"""Load `{split}_predictions.csv` from a model directory.

	- Does not parse index as dates to keep schema-agnostic behavior.
	- Returns a DataFrame as-is from CSV.
	"""
	p = Path(models_dir) / f"{split}_predictions.csv"
	if not p.exists():
		raise FileNotFoundError(str(p))
	return pd.read_csv(p)


def infer_label_column(df: pd.DataFrame) -> str:
	"""Infer the ground-truth label column from a predictions DataFrame.

	Preference order:
	  1) 'RegClass+4'
	  2) first column that is not 'pred' and does not start with 'p_'
	"""
	if "RegClass+4" in df.columns:
		return "RegClass+4"
	for c in df.columns:
		c_str = str(c)
		if c_str == "pred" or c_str.startswith("p_"):
			continue
		return c_str
	raise RuntimeError("Could not infer label column from predictions DataFrame")


def confusion_3x3_from_predictions(
	df: pd.DataFrame,
	classes: Sequence[str] = CLASSES_DEFAULT,
) -> pd.DataFrame:
	"""Compute a fixed-order 3x3 confusion matrix from a predictions DataFrame.

	Expects the DataFrame to contain a ground-truth label column (see `infer_label_column`)
	and a `pred` column with predicted class labels.

	Returns a DataFrame with index/columns equal to `classes`.
	"""
	if "pred" not in df.columns:
		raise ValueError("Predictions DataFrame must contain a 'pred' column")
	label_col = infer_label_column(df)
	y_true = df[label_col].astype(str)
	y_pred = df["pred"].astype(str)
	cm = confusion_matrix(y_true, y_pred, labels=list(classes))
	cm_df = pd.DataFrame(cm, index=list(classes), columns=list(classes))
	return cm_df


def save_confusion_matrix_csv(cm_df: pd.DataFrame, out_csv: Path, header: bool = False) -> None:
	"""Save a confusion matrix to CSV in plain numeric form by default (no headers)."""
	Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
	cm_df.to_csv(out_csv, index=False, header=header)


def classification_report_from_predictions(
	df: pd.DataFrame,
	classes: Sequence[str] = CLASSES_DEFAULT,
) -> str:
	"""Generate a text classification report for the fixed class order."""
	if "pred" not in df.columns:
		raise ValueError("Predictions DataFrame must contain a 'pred' column")
	label_col = infer_label_column(df)
	y_true = df[label_col].astype(str)
	y_pred = df["pred"].astype(str)
	return classification_report(y_true, y_pred, labels=list(classes), zero_division=0)


def compute_and_persist_confusion(
	models_dir: Path,
	split: str,
	classes: Sequence[str] = CLASSES_DEFAULT,
	out_csv_header: bool = False,
) -> pd.DataFrame:
	"""Convenience function to load predictions, compute 3x3 CM, and save CSV.

	Returns the confusion matrix DataFrame.
	"""
	df = load_predictions_csv(models_dir, split)
	cm_df = confusion_3x3_from_predictions(df, classes=classes)
	out_csv = Path(models_dir) / f"{split}_confusion_matrix.csv"
	save_confusion_matrix_csv(cm_df, out_csv, header=out_csv_header)
	return cm_df


def get_available_prediction_splits(models_dir: Path, splits: Iterable[str] = ("val", "test")) -> List[str]:
	"""Return list of splits that have a predictions CSV present in models_dir."""
	md = Path(models_dir)
	return [s for s in splits if (md / f"{s}_predictions.csv").exists()]

