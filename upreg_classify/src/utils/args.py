import argparse
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def parse_args():
    p = argparse.ArgumentParser(description='Train mFRR models (binary or multiclass).')
    p.add_argument('--task', choices=['binary', 'multiclass', 'binary_stack', 'multiclass_stack'], default='multiclass')
    p.add_argument('--output_dir', type=str, default=None, help='Directory for AutoGluon models and reports')
    p.add_argument('--time_limit', type=int, default=None, help='Time limit (seconds) for training')
    p.add_argument('--presets', type=str, default='best_quality', help='AutoGluon presets')
    p.add_argument('--model_preset', type=str, default='auto',
                   choices=['auto', 'rf_xt_only', 'rf_xt_priority', 'rf_xt_boost_stack', 'cat_focus', 'cat_only', 'cat_only_sanitized', 'xgb_only', 'xgb_only_fixed', 'lgbm_only'],
                   help='Model family configuration / prioritization')
    p.add_argument('--area', type=str, default='NO1', choices=['NO1', 'NO2'], help='Market area to train on')
    p.add_argument('--data_dir', type=str, default=os.path.join(REPO_ROOT, 'data', 'raw'))
    p.add_argument('--include_2024', action='store_true')
    p.add_argument('--no-include_2024', dest='include_2024', action='store_false')
    p.set_defaults(include_2024=True)
    p.add_argument('--heavy_interactions', action='store_true')
    p.add_argument('--dropna', action='store_true')
    p.set_defaults(dropna=True)
    p.add_argument('--train_frac', type=float, default=0.6)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--test_frac', type=float, default=0.2)
    p.add_argument('--data_start', type=str, default=None, help="Optional ISO date (YYYY-MM-DD) to start dataset from; rows earlier than this are dropped before splitting")

    # Optional preprocessing cache
    p.add_argument(
        '--preprocessed_path',
        type=str,
        default=None,
        help='Optional path to a cached preprocessed dataframe (.pkl). If present and exists, training loads it instead of recomputing preprocessing.',
    )
    p.add_argument(
        '--recompute_preprocess',
        action='store_true',
        help='If set, recompute preprocessing even if --preprocessed_path exists (and overwrite the cache).',
    )
    p.add_argument('--importance_time_limit', type=int, default=200, help='Time limit (seconds) for permutation importance')
    p.add_argument('--importance_subsample', type=int, default=1000, help='Max rows for permutation importance')
    p.add_argument('--importance_top_n', type=int, default=40, help='Top-N features to plot and print')
    p.add_argument('--activation_lag_start', type=int, default=4, help='Lag index to start retaining activation volume features from')
    p.add_argument('--single_persistence', action='store_true', help='Use single persistence model as baseline')
    # Explicit stacking / bagging controls (optional). If >0, forces stacking/bagging.
    p.add_argument('--num_bag_folds', type=int, default=0, help='Number of bagging folds (0 = AutoGluon default)')
    p.add_argument('--num_stack_levels', type=int, default=0, help='Number of stack levels (0 = AutoGluon default)')
    # Class weighting (multiclass): multiply balanced weights per class
    p.add_argument('--weight_factor_up', type=float, default=1.0, help="Multiplier for 'up' class training weight")
    p.add_argument('--weight_factor_down', type=float, default=1.0, help="Multiplier for 'down' class training weight")
    p.add_argument('--weight_factor_none', type=float, default=1.0, help="Multiplier for 'none' class training weight")
    # Decision policy tuning towards 'up'
    p.add_argument('--tune_up_bias', action='store_true', help="Enable decision-policy tuning for 'up' on validation (multiplier only)")
    p.add_argument('--tune_up_objective', type=str, default='macro', choices=['macro','up'], help="Objective for up-bias tuning: macro (macro F1) or up (F1 of up class)")
    # Volume thresholds to suppress tiny activation noise (defaults chosen as modest >0 values)
    p.add_argument('--min_up_volume', type=float, default=None, help="(Optional) Minimum Up activation volume (MW) to treat as genuine event; set to value to filter tiny activations, leave unset for no filtering")
    p.add_argument('--min_down_volume', type=float, default=None, help="(Optional) Minimum Down activation volume (MW) to treat as genuine event; set to value to filter tiny activations, leave unset for no filtering")
    # Hyperparameter tuning (currently focused on cat_only preset)
    p.add_argument('--hpo_trials', type=int, default=0, help='Number of HPO trials (if >0 enables hyperparameter tuning)')
    p.add_argument('--hpo_searcher', type=str, default='random', choices=['random','auto','bayesopt'], help='Searcher algorithm for HPO')
    p.add_argument('--hpo_scheduler', type=str, default='local', choices=['local','fifo'], help='Scheduler algorithm for HPO')
    # Feature toggles
    p.add_argument('--use_categorical_reglag', action='store_true', help='Use categorical RegLagCat-* features instead of numeric RegLag-*')
    p.add_argument('--disable_persistency_interactions', action='store_true', help='Drop features that are interactions with Persistency/PersistencyDown')
    p.add_argument('--only_persistency_features', action='store_true', help='Use only PersistenceUp/Down/None features during training')
    p.add_argument('--exclude_persistency_features', action='store_true', help='Exclude PersistenceUp/Down/None (and legacy Persistency/PersistencyDown) from features')
    return p.parse_args()