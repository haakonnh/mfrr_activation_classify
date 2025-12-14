"""
Training script for mFRR activation prediction.

Supports:
- Binary: predict Activated+4 (True/False)
- Multiclass (ternary): predict RegClass+4 ('up', 'down', 'none')

It reuses the preprocessing pipelines in src/data:
- Binary: preprocess.preprocess_all + build_feature_list
- Multiclass: data/ternary.build_multiclass_dataset

Outputs:
- AutoGluon model directory in --output_dir
- metrics.csv (summary) and classification_report.txt
- confusion matrices and predictions CSVs for val/test
"""
import os
import sys

import json
from typing import List, Optional

import numpy as np
import pandas as pd

# Import of project modules regardless of current working dir
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from autogluon.tabular import TabularPredictor

# Import preprocessing for binary
from src.data.preprocess import build_dataset, Config
from src.utils.args import parse_args
from src.train.hyperparameters import build_hyperparameters
from src.evaluation.evaluation import evaluate_and_report

###############################################################################
# paths & small utilities
###############################################################################

def _default_output_dir(task: str) -> str:
    """Default output path under repo/models/<task>_ag"""
    base = os.path.join(REPO_ROOT, 'models')
    return os.path.join(base, f'{task}_ag')


###############################################################################
# training utilities
###############################################################################

def fit_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    label: str,
    problem_type: str,
    eval_metric: str,
    output_dir: str,
    time_limit: int,
    presets: str,
    model_preset: str,
    num_bag_folds: int = 0,
    num_stack_levels: int = 0,
    hpo_trials: int = 0,
    hpo_searcher: str = 'random',
    hpo_scheduler: str = 'ASHA',
):
    """Create TabularPredictor and fit with optional hyperparameters and weights."""
    train_data = train_df[features + [label]].copy()
    val_data = val_df[features + [label]].copy() if len(val_df) else None

    predictor = TabularPredictor(
        label=label,
        path=output_dir,
        problem_type=problem_type,
        eval_metric=eval_metric,
        sample_weight="auto_weight"
    )

    fit_kwargs = dict(
        train_data=train_data,
        tuning_data=val_data,
        time_limit=time_limit,
        presets=presets,
        verbosity=2,
    )

    # AutoGluon decision-threshold calibration is only valid for binary tasks
    # Ref: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
    if problem_type == 'binary':
        fit_kwargs['calibrate_decision_threshold'] = True

    # Model family selection
    hyperparameters = build_hyperparameters(model_preset, hpo_trials=hpo_trials)
    if hyperparameters is not None:
        fit_kwargs['hyperparameters'] = hyperparameters

    # Attach HPO configuration if requested
    if hpo_trials and hpo_trials > 0:
        fit_kwargs['hyperparameter_tune_kwargs'] = {
            'num_trials': int(hpo_trials),
            'searcher': hpo_searcher,
            'scheduler': hpo_scheduler,
        }

    if val_data is not None and isinstance(num_bag_folds, int) and num_bag_folds > 0:
        # Only needed when passing tuning_data in bagged modes
        fit_kwargs['use_bag_holdout'] = True

    # Explicitly control stacking/bagging. Important: even zeros should override preset defaults
    # to avoid unintended bagging/stacking during HPO or fast runs.
    if isinstance(num_bag_folds, int):
        fit_kwargs['num_bag_folds'] = int(num_bag_folds)
    if isinstance(num_stack_levels, int):
        fit_kwargs['num_stack_levels'] = int(num_stack_levels)
    #fit_kwargs["auto_stack"] = True
    predictor.fit(**fit_kwargs)
    return predictor

###############################################################################
# top level 
###############################################################################

def train_and_evaluate(
    task: str,
    output_dir: str,
    time_limit: int,
    presets: str,
    model_preset: str,
    area: str,
    data_dir: str,
    include_2024: bool,
    heavy_interactions: bool,
    dropna: bool,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    importance_time_limit: int = 30,
    importance_subsample: int = 8000,
    importance_top_n: int = 40,
    activation_lag_start: int = 4,
    single_persistence: bool = False,
    weight_factor_up: float = 1.0,
    weight_factor_down: float = 1.0,
    weight_factor_none: float = 1.0,
    tune_up_bias: bool = True,
    num_bag_folds: int = 0,
    num_stack_levels: int = 0,
    min_up_volume: Optional[float] = None,
    min_down_volume: Optional[float] = None,
    tune_up_objective: str = 'up',
    hpo_trials: int = 0,
    hpo_searcher: str = 'random',
    hpo_scheduler: str = 'local',
    use_categorical_reglag: bool = False,
    data_start: Optional[str] = None,
    disable_persistency_interactions: bool = False,
    only_persistency_features: bool = False,
    exclude_persistency_features: bool = False,
    preprocessed_path: Optional[str] = None,
    force_recompute_preprocess: bool = False,
):
        
    # 1) Build dataset
    # Set default values
    #if not num_bag_folds:
        #num_bag_folds = 0
    #if not num_stack_levels:
        #num_stack_levels = 0
    
    cfg = Config(
        data_dir=data_dir,
        area=area,
        include_2024=include_2024,
        heavy_interactions=heavy_interactions,
        dropna=dropna,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        activation_lag_start=activation_lag_start,
        single_persistence=single_persistence,
        min_up_volume=min_up_volume,
        min_down_volume=min_down_volume,
    )

    # Build dataset using label RegClass+4
    label = "RegClass+4"
    df, (train_df, val_df, test_df), features = build_dataset(
        cfg,
        label_name=label,
        preprocessed_path=preprocessed_path,
        force_recompute=force_recompute_preprocess,
    )
    #features = [f for f in features if f not in ('Activated', 'RegClass')]
    problem_type, eval_metric = 'multiclass', 'f1_macro'

    # Optional: restrict dataset to rows starting at data_start (inclusive)
    if data_start:
        try:
            start_ts = pd.Timestamp(data_start)
        except Exception as e:
            raise ValueError(f"Invalid --data_start value '{data_start}': {e}")
        
        df_filtered = df[df.index >= start_ts].copy()
        if len(df_filtered) == 0:
            raise ValueError(f"After applying --data_start={data_start}, dataset is empty.")
        # Recompute splits on filtered dataset using provided fractions
        n = len(df_filtered)
        t_end = int(n * train_frac)
        v_end = t_end + int(n * val_frac)
        t_end = max(min(t_end, n), 0)
        v_end = max(min(v_end, n), t_end)
        train_df = df_filtered.iloc[:t_end].copy()
        val_df = df_filtered.iloc[t_end:v_end].copy()
        test_df = df_filtered.iloc[v_end:].copy() if test_frac > 0 else df_filtered.iloc[v_end:v_end].copy()
        df = df_filtered
        print(f"Applied dataset start filter: from {data_start} -> rows: {len(df)} (train {len(train_df)}, val {len(val_df)}, test {len(test_df)})")
        print(f"Last timestamp in dataset: {df.index.max()}")
    # Optionally swap numeric RegLag-* for categorical RegLagCat-* in the feature set
    if use_categorical_reglag:
        # Identify available lags
        reglag_nums = [c for c in features if c.startswith('RegLag-')]
        reglag_cats = [c for c in df.columns if c.startswith('RegLagCat-')]
        # Remove numeric
        features = [c for c in features if c not in reglag_nums]
        # Add categorical if present
        for c in reglag_cats:
            if c not in features and c in df.columns:
                features.append(c)

    # Optional: disable interaction features with Persistency (before training)
    if disable_persistency_interactions:
        # Remove any feature that includes Persistency-based interactions while keeping base Persistency columns
        def _keep_feature(name: str) -> bool:
            patterns = [
                ' x Persistency', 'Persistency x ',
                ' x PersistencyDown', 'PersistencyDown x ',
                ' x PersistenceUp', 'PersistenceUp x ',
                ' x PersistenceDown', 'PersistenceDown x ',
                ' x PersistenceNone', 'PersistenceNone x ',
            ]
            return not any(p in name for p in patterns)
        features = [f for f in features if _keep_feature(f)]

    # Optional: restrict to Persistency-only features, overriding all others
    if only_persistency_features:
        allowed = [c for c in ['PersistenceUp', 'PersistenceDown', 'PersistenceNone'] if c in df.columns]
        if not allowed:
            raise ValueError('Persistency-only flag enabled, but no Persistence* columns exist in dataset.')
        features = allowed
        # Trim frames to just allowed features + label
        keep_cols = features + [label]
        train_df = train_df[keep_cols]
        val_df = val_df[keep_cols]
        test_df = test_df[keep_cols]
    elif exclude_persistency_features:
        persistency_cols = {'PersistenceUp','PersistenceDown','PersistenceNone','Persistency','PersistencyDown'}
        features = [c for c in features if c not in persistency_cols]

    # 2) Output dirs
    os.makedirs(output_dir, exist_ok=True)
    reports_fig_dir = os.path.join(REPO_ROOT, 'reports', 'figures', task)
    os.makedirs(reports_fig_dir, exist_ok=True)
    reports_df_dir = os.path.join(REPO_ROOT, 'reports', 'dataframes')
    os.makedirs(reports_df_dir, exist_ok=True)

    # 3) Fit
    predictor = fit_predictor(
        train_df, val_df, features, label, problem_type, eval_metric,
        output_dir, time_limit, presets, model_preset,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        hpo_trials=hpo_trials,
        hpo_searcher=hpo_searcher,
        hpo_scheduler=hpo_scheduler,
    )

    # 4) Evaluate & report
    metrics = evaluate_and_report(
        predictor, train_df, val_df, test_df, features, label, problem_type, output_dir,
        reports_fig_dir, reports_df_dir, importance_time_limit, importance_subsample, importance_top_n,
        tune_up_bias=tune_up_bias,
        tune_up_objective=tune_up_objective,
    )

    print('Training complete.')
    print('Metrics:', json.dumps(metrics, indent=2))
    print('Model path:', output_dir)

    # Note: persistency-interaction feature filtering is applied before training above.

if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir(args.task)

    train_and_evaluate(
        task=args.task,
        output_dir=output_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        model_preset=args.model_preset,
        area=args.area,
        data_dir=args.data_dir,
        include_2024=args.include_2024,
        heavy_interactions=args.heavy_interactions,
        dropna=args.dropna,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        importance_time_limit=args.importance_time_limit,
        importance_subsample=args.importance_subsample,
        importance_top_n=args.importance_top_n,
        single_persistence=args.single_persistence,
        weight_factor_up=args.weight_factor_up,
        weight_factor_down=args.weight_factor_down,
        weight_factor_none=args.weight_factor_none,
        tune_up_bias=args.tune_up_bias,
        tune_up_objective=args.tune_up_objective,
        num_bag_folds=args.num_bag_folds,
        num_stack_levels=args.num_stack_levels,
        min_up_volume=args.min_up_volume,
        min_down_volume=args.min_down_volume,
        hpo_trials=args.hpo_trials,
        hpo_searcher=args.hpo_searcher,
        hpo_scheduler=args.hpo_scheduler,
        use_categorical_reglag=args.use_categorical_reglag,
        data_start=args.data_start,
        disable_persistency_interactions=args.disable_persistency_interactions,
        only_persistency_features=args.only_persistency_features,
        exclude_persistency_features=args.exclude_persistency_features,
        preprocessed_path=args.preprocessed_path,
        force_recompute_preprocess=args.recompute_preprocess,
    )
