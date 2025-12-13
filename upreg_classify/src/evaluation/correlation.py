r"""
Correlation, clustering, VIF, PCA and permutation/SHAP importance
for a trained AutoGluon predictor.

Usage (Windows cmd):
  c:\PythonProjects\rl_reserve_markets\.venv1\Scripts\python.exe \
	upreg_classify\src\evaluation\correlation.py \
	--models-dir c:\PythonProjects\rl_reserve_markets\upreg_classify\models\quick_exclude_persistency_check \
	--max-rows 5000

Outputs:
  - Figures saved under `upreg_classify/reports/figures/multiclass`:
	  correlation_cluster.png, pca_explained_variance.png
  - CSVs saved under `upreg_classify/reports/dataframes`:
	  permutation_importance.csv, shap_importance.csv
  - Console prints: top clusters, VIF table head, permutation/shap top features
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Tuple

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Optional deps, guard imports
try:
	from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
	variance_inflation_factor = None

try:
	from sklearn.decomposition import PCA
	from sklearn.inspection import permutation_importance
	from sklearn.metrics import make_scorer, f1_score
except Exception:
	PCA = None
	permutation_importance = None
	make_scorer = None
	f1_score = None

import importlib.util as _importlib_util
_shap_available = _importlib_util.find_spec('shap') is not None


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_FIG_DIR = REPO_ROOT / 'upreg_classify' / 'reports' / 'figures' / 'multiclass'
REPORTS_DF_DIR = REPO_ROOT / 'upreg_classify' / 'reports' / 'dataframes'


def _load_predictor(models_dir: Path):
	from autogluon.tabular import TabularPredictor
	return TabularPredictor.load(models_dir)


def _load_dataset_for_features(predictor, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
	"""Load a DataFrame containing at least the predictor.features() + label.
	Prefers a cached dataset snapshot if present; otherwise uses the predictor's internal data.
	Returns (X, df_all, label).
	"""
	label = predictor.label
	# Prefer cached dataset snapshot
	df_path = REPORTS_DF_DIR / 'multiclass_dataset.csv'
	df_all: Optional[pd.DataFrame] = None
	if df_path.exists():
		df_all = pd.read_csv(df_path)
	else:
		# Fallback to internal training data
		try:
			df_all = predictor.load_data_internal('train')[0]
		except Exception:
			raise RuntimeError('Could not load dataset snapshot or internal training data.')

	# Restrict to model features
	feature_cols = [c for c in df_all.columns if c in predictor.features()]
	# Dropna only on features; label may be absent in snapshot
	df_all = df_all.dropna(subset=feature_cols).copy()
	if max_rows and len(df_all) > max_rows:
		df_all = df_all.tail(max_rows).copy()
	X = df_all[feature_cols].copy()
	# Ensure label exists in DataFrame, else return None
	lbl = label if label in df_all.columns else None
	return X, df_all, lbl


def compute_correlation_clusters(X: pd.DataFrame, fig_out: Path) -> Tuple[pd.Series, pd.DataFrame]:
	"""Return (cluster_map, top_pairs_df) and save a clustered heatmap.
	Uses condensed distance matrix to avoid SciPy warnings.
	"""
	# Numeric-only for correlation
	Xn = X.select_dtypes(include=[np.number]).copy()
	if Xn.empty:
		print('No numeric features available for correlation.')
		return pd.Series(dtype=int), pd.DataFrame()
	corr = Xn.corr().abs()
	# Plot cluster map
	try:
		sns.clustermap(corr, figsize=(10, 10), cmap='viridis')
		plt.title('Feature correlation (abs) with hierarchical clustering')
		plt.savefig(fig_out, bbox_inches='tight')
		plt.close()
	except Exception as e:
		print('Correlation heatmap failed:', e)
	# Hierarchical clustering (use condensed distance)
	dist = 1 - corr.values
	np.fill_diagonal(dist, 0.0)
	Z = linkage(squareform(dist), method='average')
	clusters = fcluster(Z, t=0.6, criterion='distance')
	cluster_map = pd.Series(clusters, index=corr.columns, name='cluster')
	# Top correlated pairs
	pairs = (
		corr.stack()
		.reset_index()
		.rename(columns={'level_0': 'feature_a', 'level_1': 'feature_b', 0: 'corr'})
	)
	pairs = pairs[pairs['feature_a'] < pairs['feature_b']]
	top_pairs = pairs.sort_values('corr', ascending=False)
	return cluster_map, top_pairs


def compute_vif(X: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
	if variance_inflation_factor is None:
		print('statsmodels not available; skipping VIF.')
		return pd.DataFrame()
	Xn = X.select_dtypes(include=[np.number]).copy()
	if Xn.empty:
		print('No numeric features available for VIF.')
		return pd.DataFrame()
	# Standardize
	X_std = (Xn - Xn.mean()) / (Xn.std(ddof=0).replace(0, np.nan))
	X_std = X_std.fillna(0.0)
	vif_vals = []
	for i in range(X_std.shape[1]):
		try:
			vif_vals.append(variance_inflation_factor(X_std.values, i))
		except Exception:
			vif_vals.append(np.nan)
	vif = pd.Series(vif_vals, index=X_std.columns)
	df = pd.DataFrame({'feature': X_std.columns, 'VIF': vif.values})
	def _bucket(v: float) -> str:
		if np.isinf(v):
			return 'infinite'
		if pd.isna(v):
			return 'nan'
		if v >= 10:
			return 'high(>=10)'
		if v >= 5:
			return 'moderate(5-10)'
		return 'low(<5)'
	df['bucket'] = df['VIF'].apply(_bucket)
	df = df.sort_values('VIF', ascending=False)
	# Console summary
	counts = df['bucket'].value_counts().to_dict()
	print('VIF buckets:', counts)
	print('Top (by VIF):')
	print(df.head(top_n).to_string(index=False))
	return df


def compute_pca_plot(X: pd.DataFrame, fig_out: Path) -> Optional[np.ndarray]:
	if PCA is None:
		print('sklearn PCA not available; skipping PCA.')
		return None
	Xn = X.select_dtypes(include=[np.number]).copy()
	if Xn.empty:
		print('No numeric features available for PCA.')
		return None
	X_std = (Xn - Xn.mean()) / (Xn.std(ddof=0).replace(0, np.nan))
	X_std = X_std.fillna(0.0)
	pca = PCA().fit(X_std)
	evr = pca.explained_variance_ratio_
	csum = np.cumsum(evr)
	plt.figure(figsize=(8, 4))
	plt.plot(csum)
	plt.xlabel('n components')
	plt.ylabel('cumulative explained variance')
	plt.title('PCA explained variance')
	plt.tight_layout()
	plt.savefig(fig_out, bbox_inches='tight')
	plt.close()
	# Console readout: components to reach 90%
	k90 = int(np.searchsorted(csum, 0.90) + 1)
	print(f'PCA: components to reach 90% variance: {k90}')
	return evr


class _PredictorWrapper:
	def __init__(self, predictor):
		self.predictor = predictor
	def predict(self, X):
		return self.predictor.predict(X)


def compute_permutation_importance(predictor, X: pd.DataFrame, y: pd.Series, out_csv: Path, n_repeats: int = 10, max_rows: int = 1000) -> pd.Series:
	if permutation_importance is None or make_scorer is None or f1_score is None:
		print('sklearn not available; skipping permutation importance.')
		return pd.Series(dtype=float)
	# Sample rows for speed
	df = X.copy()
	df['__y__'] = y.values
	if len(df) > max_rows:
		df = df.sample(n=max_rows, random_state=0)
	Xs = df.drop(columns=['__y__'])
	ys = df['__y__']
	scorer = make_scorer(f1_score, average='macro')
	model = _PredictorWrapper(predictor)
	res = permutation_importance(model, Xs, ys, n_repeats=n_repeats, random_state=0, scoring=scorer)
	imp = pd.Series(res.importances_mean, index=Xs.columns).sort_values(ascending=False)
	try:
		out_csv.parent.mkdir(parents=True, exist_ok=True)
		imp.to_csv(out_csv, index=True)
	except Exception as e:
		print('Saving permutation importance failed:', e)
	print('Top permutation importance features:')
	print(imp.head(30).to_string())
	return imp


def compute_shap_importance(predictor, X: pd.DataFrame, out_csv: Path, max_rows: int = 2000) -> pd.Series:
	if not _shap_available:
		print('shap not available; skipping SHAP importance.')
		return pd.Series(dtype=float)
	# Sample rows
	Xs = X
	if len(Xs) > max_rows:
		Xs = Xs.sample(n=max_rows, random_state=0)
	# Try to access underlying CatBoost model
	model_obj = None
	try:
		trainer = predictor._trainer  # private; guard
		best_name = trainer.get_model_best()
		m = trainer.load_model(best_name)
		# AutoGluon CatBoost model has .model holding catboost.CatBoostClassifier
		model_obj = getattr(m, 'model', None)
	except Exception:
		model_obj = None
	if model_obj is None:
		print('Underlying CatBoost model not accessible; skipping SHAP.')
		return pd.Series(dtype=float)
	# Import shap lazily to avoid editor import warnings
	import shap  # type: ignore
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		explainer = shap.TreeExplainer(model_obj)
		shap_vals = explainer.shap_values(Xs)
	# Multiclass: list of arrays; aggregate mean abs across classes
	if isinstance(shap_vals, list):
		# shape: [n_classes][n_rows, n_features]
		try:
			shap_abs = np.mean([np.abs(v) for v in shap_vals], axis=0)
		except Exception:
			# Fallback: just take first class
			shap_abs = np.abs(shap_vals[0])
	else:
		shap_abs = np.abs(shap_vals)
	shap_mean = np.mean(shap_abs, axis=0)
	imp = pd.Series(shap_mean, index=Xs.columns).sort_values(ascending=False)
	try:
		out_csv.parent.mkdir(parents=True, exist_ok=True)
		imp.to_csv(out_csv, index=True)
	except Exception as e:
		print('Saving SHAP importance failed:', e)
	print('Top SHAP mean|abs| features:')
	print(imp.head(30).to_string())
	return imp


def main(models_dir: str, max_rows: int = 5000):
	models_path = Path(models_dir)
	assert models_path.exists(), f'Models path not found: {models_path}'
	REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
	REPORTS_DF_DIR.mkdir(parents=True, exist_ok=True)

	predictor = _load_predictor(models_path)
	X, df_all, label = _load_dataset_for_features(predictor, max_rows=max_rows)
	y = df_all[label].astype(str) if label else None

	print('\n=== Correlation & Clusters ===')
	corr_fig = REPORTS_FIG_DIR / 'correlation_cluster.png'
	cluster_map, top_pairs = compute_correlation_clusters(X, corr_fig)
	if len(cluster_map):
		# Save cluster map CSV
		cluster_csv = REPORTS_DF_DIR / 'feature_clusters.csv'
		cluster_csv.parent.mkdir(parents=True, exist_ok=True)
		cluster_map.to_csv(cluster_csv, header=True)
		# Print largest clusters
		groups = (
			cluster_map.reset_index().rename(columns={'index': 'feature'})
			.groupby(cluster_map.values).agg({'feature': list})
			.reset_index().rename(columns={'index': 'cluster', 0: 'cluster'})
		)
		groups['size'] = groups['feature'].apply(len)
		groups = groups.sort_values('size', ascending=False)
		print('Largest clusters:')
		for _, row in groups.head(5).iterrows():
			print(f"  cluster {int(row[cluster_map.name])} (n={row['size']}): {', '.join(row['feature'][:10])}{' ...' if row['size']>10 else ''}")
	# Top correlated pairs
	if not top_pairs.empty:
		high = top_pairs[top_pairs['corr'] >= 0.8].copy()
		print('Top correlated pairs (abs>=0.8):')
		for _, r in high.head(20).iterrows():
			print(f"  {r['feature_a']} ~ {r['feature_b']}: {r['corr']:.3f}")
		# Save CSV
		(REPORTS_DF_DIR / 'corr_top_pairs.csv').parent.mkdir(parents=True, exist_ok=True)
		high.to_csv(REPORTS_DF_DIR / 'corr_top_pairs.csv', index=False)

	print('\n=== Multicollinearity (VIF) ===')
	vif_df = compute_vif(X, top_n=20)
	if not vif_df.empty:
		vif_out = REPORTS_DF_DIR / 'vif.csv'
		vif_out.parent.mkdir(parents=True, exist_ok=True)
		vif_df.to_csv(vif_out, index=False)

	print('\n=== PCA Explained Variance ===')
	pca_fig = REPORTS_FIG_DIR / 'pca_explained_variance.png'
	evr = compute_pca_plot(X, pca_fig)
	if evr is not None:
		pca_csv = REPORTS_DF_DIR / 'pca_evr.csv'
		pca_csv.parent.mkdir(parents=True, exist_ok=True)
		pd.DataFrame({'explained_variance_ratio': evr}).to_csv(pca_csv, index=False)

	print('\n=== Permutation Importance (sklearn) ===')
	if y is not None:
		perm_csv = REPORTS_DF_DIR / 'permutation_importance.csv'
		compute_permutation_importance(predictor, X, y, perm_csv, n_repeats=10, max_rows=1000)
	else:
		print('Skipped: label not present in dataset snapshot.')

	print('\n=== SHAP Importance (CatBoost) ===')
	shap_csv = REPORTS_DF_DIR / 'shap_importance.csv'
	compute_shap_importance(predictor, X, shap_csv, max_rows=2000)

	# Final summary
	print('\n=== Outputs ===')
	print(f"Figures: {REPORTS_FIG_DIR}")
	print(f"CSVs:    {REPORTS_DF_DIR}")


if __name__ == '__main__':
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--models-dir', type=str, default=str(REPORTS_FIG_DIR.parent.parent / 'models' / 'quick_exclude_persistency_check'))
	ap.add_argument('--max-rows', type=int, default=5000)
	args = ap.parse_args()
	main(models_dir=args.models_dir, max_rows=args.max_rows)