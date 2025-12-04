"""
Plotting utilities for model analysis.

Includes feature importance computation via AutoGluon TabularPredictor
and a simple horizontal bar plot helper.
"""
from __future__ import annotations
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_feature_importance_ag(
	predictor,
	data: pd.DataFrame,
	features: list[str],
	label: str,
	subsample_size: int | None = 10000,
	time_limit: int | None = 60,
) -> pd.DataFrame:
	"""
	Compute permutation-based feature importance using AutoGluon.

	Parameters
	- predictor: AutoGluon TabularPredictor
	- data: DataFrame containing features and label
	- features: list of feature column names
	- label: label column name
	- subsample_size: optional subsample size to speed up importance
	- time_limit: optional time budget in seconds

	Returns a DataFrame with at least ['importance'] indexed by feature (AG format).
	"""
	cols = [c for c in features if c in data.columns]
	if label in data.columns:
		cols = cols + [label]
	df_eval = data[cols].copy()
	imp_df = predictor.feature_importance(
		df_eval,
		subsample_size=min(subsample_size, len(df_eval)) if subsample_size else None,
		time_limit=time_limit,
	)
	# Ensure a flat DF with 'feature' column for plotting convenience
	if 'importance' in imp_df.columns and 'index' in imp_df.columns:
		imp_df = imp_df.rename(columns={'index': 'feature'})
	elif imp_df.index.name is not None or not 'feature' in imp_df.columns:
		imp_df = imp_df.reset_index().rename(columns={'index': 'feature'})
	return imp_df


def plot_feature_importance(
	imp_df: pd.DataFrame,
	top_n: int = 30,
	figsize: Tuple[float, float] | None = None,
	output_path: str | None = None,
):
	"""
	Plot top-N features by importance as a horizontal bar chart.

	Returns (fig, ax) for further customization.
	"""
	if 'feature' not in imp_df.columns:
		imp_df = imp_df.reset_index().rename(columns={'index': 'feature'})
	if 'importance' not in imp_df.columns:
		# Try common alternative column names
		for alt in ['importance_mean', 'score']:
			if alt in imp_df.columns:
				imp_df = imp_df.rename(columns={alt: 'importance'})
				break
	top = imp_df.sort_values('importance', ascending=False).head(top_n)
	if figsize is None:
		figsize = (8, max(4, 0.3 * len(top)))
	fig, ax = plt.subplots(figsize=figsize)
	sns.barplot(data=top, x='importance', y='feature', color='#4472C4', ax=ax)
	ax.set_xlabel('Importance')
	ax.set_ylabel('Feature')
	ax.set_title(f'Top {len(top)} Feature Importances')
	plt.tight_layout()
	if output_path:
		fig.savefig(output_path, dpi=150)
	return fig, ax

