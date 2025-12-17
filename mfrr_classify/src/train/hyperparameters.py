from autogluon.common import space as ag_space  # For sanitized CatBoost HPO search space (AutoGluon 1.4.x)

###############################################################################
# hyperparameters selection
###############################################################################

def build_hyperparameters(mode: str, hpo_trials: int = 0):
    """Return model family hyperparameters (including sanitized CatBoost HPO space when applicable).

    Modes:
      - 'auto': Defer entirely to AutoGluon presets (return None)
      - 'rf_xt_only': Only RF + XT variants
      - 'rf_xt_priority': Full mix, RF/XT boosted priority
      - 'rf_xt_boost_stack': Compact RF/XT + boosted boosting families for stacking
      - 'cat_focus': Several curated CatBoost variants + small XT
      - 'cat_only': Single CatBoost family; if hpo_trials>0, returns sanitized HPO search space
      - 'cat_only_sanitized': Explicit sanitized CatBoost HPO space (alias; same as cat_only with hpo)
    """
    if mode == 'auto':
        return None

    rf_variants = [
        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini'}},
        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr'}},
        {'n_estimators': 400, 'max_features': 0.8, 'ag_args': {'name_suffix': 'N400'}},
        {'n_estimators': 800, 'max_features': 0.6, 'ag_args': {'name_suffix': 'N800'}},
    ]
    xt_variants = [
        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini'}},
        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr'}},
        {'n_estimators': 800, 'max_features': 0.6, 'ag_args': {'name_suffix': 'N800'}},
    ]
    if mode == 'rf_xt_only':
        return {'RF': rf_variants, 'XT': xt_variants}

    if mode == 'rf_xt_priority':
        rf_boost = [{**cfg, 'ag_args': {**cfg.get('ag_args', {}), 'priority': 100}} for cfg in rf_variants]
        xt_boost = [{**cfg, 'ag_args': {**cfg.get('ag_args', {}), 'priority': 100}} for cfg in xt_variants]
        base = {
            'GBM': [{}], 'CAT': [{}], 'XGB': [{}], 'FASTAI': [{}], 'NN_TORCH': [{}],
        }
        return {'RF': rf_boost, 'XT': xt_boost, **base}

    if mode == 'rf_xt_boost_stack':
        rf_core = [{'n_estimators': 300, 'max_features': 0.8, 'ag_args': {'name_suffix': 'Core', 'priority': 90}}]
        xt_core = [{'n_estimators': 400, 'max_features': 0.7, 'ag_args': {'name_suffix': 'Core', 'priority': 90}}]
        boosting = {
            'GBM': [{'ag_args': {'priority': 110}}],
            'CAT': [{'ag_args': {'priority': 110}}],
            'XGB': [{'ag_args': {'priority': 100}}],
        }
        others = {'FASTAI': [{'ag_args': {'priority': 10}}], 'NN_TORCH': [{'ag_args': {'priority': 10}}]}
        return {'RF': rf_core, 'XT': xt_core, **boosting, **others}

    if mode == 'cat_focus':
        cat_variants = [
            {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.0,
             'ag_args': {'name_suffix': 'D6lr0.1', 'priority': 130}},
            {'depth': 8, 'learning_rate': 0.06, 'l2_leaf_reg': 5.0, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.5,
             'ag_args': {'name_suffix': 'D8lr0.06', 'priority': 125}},
            {'depth': 10, 'learning_rate': 0.03, 'l2_leaf_reg': 8.0, 'bootstrap_type': 'MVS',
             'ag_args': {'name_suffix': 'D10lr0.03', 'priority': 120}},
            {'depth': 6, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.5,
             'ag_args': {'name_suffix': 'D6lr0.2', 'priority': 115}},
        ]
        xt_light = [{'n_estimators': 300, 'max_features': 0.7, 'ag_args': {'name_suffix': 'Light', 'priority': 30}}]
        return {'CAT': cat_variants, 'XT': xt_light}

    if mode in ('cat_only', 'cat_only_sanitized'):
        if hpo_trials and hpo_trials > 0:
            return {
                'CAT': [{
                    'depth': ag_space.Int(4, 8),
                    'learning_rate': ag_space.Real(0.02, 0.2),
                    'l2_leaf_reg': ag_space.Real(1.0, 14.0),
                    'subsample': ag_space.Real(0.6, 1.0),
                    'colsample_bylevel': ag_space.Real(0.4, 1.0),
                    'bootstrap_type': 'Bernoulli',
                    'random_strength': ag_space.Real(0.0, 2.0),
                    'iterations': 2000,
                }]
            }
        return {'CAT': [{}]}

    if mode == 'xgb_only':
        # If HPO trials requested, return sanitized XGBoost search space.
        if hpo_trials and hpo_trials > 0:
            return {
                'XGB': [{
                    'max_depth': ag_space.Int(3, 9),
                    'learning_rate': ag_space.Real(0.01, 0.2),
                    'subsample': ag_space.Real(0.6, 1.0),
                    'colsample_bytree': ag_space.Real(0.4, 1.0),
                    'gamma': ag_space.Real(0.0, 5.0),
                    'min_child_weight': ag_space.Int(1, 8),
                    'reg_alpha': ag_space.Real(0.0, 2.0),
                    'reg_lambda': ag_space.Real(0.5, 5.0),
                    'n_estimators': 1500,
                }]
            }
        return {'XGB': [{}]}

    if mode == 'xgb_only_fixed':
        # Stable set of fixed XGBoost configs to emulate a small search without HPO
        xgb_variants = [
            {'n_estimators': 1200, 'max_depth': 6, 'learning_rate': 0.06, 'subsample': 0.9, 'colsample_bytree': 0.8,
             'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_weight': 1, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D6lr0.06'}},
            {'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7,
             'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_weight': 3, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D8lr0.05'}},
            {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.9,
             'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_weight': 1, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D5lr0.1'}},
            {'n_estimators': 1500, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7,
             'reg_alpha': 0.0, 'reg_lambda': 2.0, 'min_child_weight': 5, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D7lr0.03'}},
            {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0,
             'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_weight': 1, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D4lr0.2'}},
            {'n_estimators': 1500, 'max_depth': 9, 'learning_rate': 0.04, 'subsample': 0.75, 'colsample_bytree': 0.6,
             'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 6, 'gamma': 0.0,
             'ag_args': {'name_suffix': 'D9lr0.04L1_0.5L2_2'}},
        ]
        return {'XGB': xgb_variants}

    if mode == 'lgbm_only':
        # LightGBM-only family; if HPO trials requested, return sanitized search space
        if hpo_trials and hpo_trials > 0:
            return {
                'GBM': [{
                    # Core tree shape + regularization
                    'num_leaves': ag_space.Int(16, 128),
                    'learning_rate': ag_space.Real(0.01, 0.2),
                    'feature_fraction': ag_space.Real(0.6, 1.0),  # colsample
                    'bagging_fraction': ag_space.Real(0.6, 1.0),  # subsample
                    'bagging_freq': ag_space.Int(0, 10),
                    'min_data_in_leaf': ag_space.Int(10, 200),
                    'lambda_l1': ag_space.Real(0.0, 2.0),
                    'lambda_l2': ag_space.Real(0.0, 4.0),
                    'num_boost_round': 1500,
                }]
            }
        return {'GBM': [{}]}

    return None


