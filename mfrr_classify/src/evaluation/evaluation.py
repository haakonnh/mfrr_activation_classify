import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Optional

from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from src.utils.plotting import compute_feature_importance_ag, plot_feature_importance

def _predict_labels_with_policy(
    predictor: TabularPredictor,
    X: pd.DataFrame,
    policy: Optional[Dict[str, float | str]] = None,
) -> pd.Series:
    """Predict class labels under an optional decision policy.

        Supported policy (multiclass):
        - {'type': 'multiplier', 'up': alpha}
            Multiplies p('up') by alpha prior to argmax.
    """
    if not policy:
        return predictor.predict(X)

    proba = predictor.predict_proba(X)
    if not isinstance(proba, pd.DataFrame):
        return predictor.predict(X)

    kind = policy.get('type')
    if kind == 'multiplier':
        adj = proba.copy()
        for cls, m in policy.items():
            if cls == 'type':
                continue
            if cls in adj.columns and m is not None:
                try:
                    adj[str(cls)] = adj[str(cls)] * float(m)
                except (TypeError, ValueError):
                    pass
        return adj.idxmax(axis=1)
    else:
        return predictor.predict(X)
    
def _tune_up_multiplier_obj(
    predictor: TabularPredictor,
    val_df: pd.DataFrame,
    features: List[str],
    label: str,
    candidates: List[float] = [0.75, 0.9, 1, 1.1, 1.175, 1.25, 1.3, 1.375, 1.45, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4, 5.0],
    optimize_for: str = 'up',  # 'macro' | 'up'
    verbose: bool = False,
) -> tuple[float, Dict[str, float | str]]:
    """Grid-search a bias multiplier for 'up'. Optimizes for macro F1 by default.

    Returns (best_alpha, policy) where policy={'type': 'multiplier', 'up': alpha}.
    """
    
    optimize_for = str(optimize_for or 'up').lower()
    if optimize_for not in {'macro', 'up'}:
        optimize_for = 'up'
    if val_df is None or len(val_df) == 0:
        return 1.0, {}
    X = val_df[features]
    y_true = val_df[label]
    proba = predictor.predict_proba(X)
    if not isinstance(proba, pd.DataFrame) or 'up' not in proba.columns:
        return 1.0, {}
    best_alpha, best_obj = 1.0, -1.0
    for alpha in candidates:
        adj = proba.copy()
        adj['up'] = adj['up'] * alpha
        y_pred = adj.idxmax(axis=1)
        f1_up = f1_score((y_true == 'up'), (y_pred == 'up'), zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if verbose:
            print(f'Tuning up-multiplier alpha={alpha:.2f}: F1(up)={f1_up:.4f}, F1(macro)={f1_macro:.4f}')
        obj = f1_macro if optimize_for == 'macro' else f1_up
        if obj > best_obj:
            best_obj, best_alpha = obj, alpha
    obj_name = 'f1_macro' if optimize_for == 'macro' else 'f1_up'
    print(f'Selected up-multiplier alpha={best_alpha:.2f} ({obj_name}={best_obj:.4f})')
    return float(best_alpha), {'type': 'multiplier', 'up': float(best_alpha)}





def evaluate_and_report(
    predictor: TabularPredictor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label: str,
    problem_type: str,
    output_dir: str,
    reports_fig_dir: str,
    reports_df_dir: str,
    importance_time_limit: int,
    importance_subsample: int,
    importance_top_n: int,
    tune_up_bias: bool = True,
    tune_up_objective: str = 'up',  # 'macro' or 'up'
):
    """Run evaluation on val/test, save artifacts, compute feature importance."""
    # Persist dataset snapshot for notebooks
    os.makedirs(reports_df_dir, exist_ok=True)
    task = 'multiclass' if problem_type == 'multiclass' else 'binary'
    train_df.to_csv(os.path.join(reports_df_dir, f'{task}_dataset.csv'), index=False)

    # Optional: favor 'up' via tuned policy on validation
    policy: Dict[str, float | str] | None = None
    if tune_up_bias and problem_type == 'multiclass':
        eval_base = val_df if len(val_df) else train_df
        alpha, policy = _tune_up_multiplier_obj(
            predictor,
            eval_base,
            features,
            label,
            optimize_for=tune_up_objective,
            verbose=False,
        )
        with open(os.path.join(output_dir, 'class_multipliers.json'), 'w', encoding='utf-8') as f:
            json.dump({'up': alpha}, f, indent=2)

    metrics: dict = {}
    printed_artifacts: Dict[str, Dict[str, object]] = {}
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        if len(split_df) == 0:
            continue
        X = split_df[features]
        y_true = split_df[label]
        y_pred = _predict_labels_with_policy(predictor, X, policy)

        if problem_type == 'multiclass':
            f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            metrics[f'{split_name}_f1_macro'] = f1m
            metrics[f'{split_name}_accuracy'] = acc
        else:
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            metrics[f'{split_name}_f1'] = f1
            metrics[f'{split_name}_accuracy'] = acc

        rep = classification_report(y_true, y_pred, zero_division=0)
        with open(os.path.join(output_dir, f'{split_name}_classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(rep)
        # Ensure a fixed 3x3 confusion matrix for the ternary setup.
        # Prefer the probability dataframe's columns as canonical class order.
        labels_to_use: List[str] | None = None
        if problem_type == 'multiclass':
            preferred = ['up', 'down', 'none']
            try:
                _proba_cols = predictor.predict_proba(X)
                if isinstance(_proba_cols, pd.DataFrame):
                    cols = [str(c) for c in _proba_cols.columns]
                    if set(preferred).issubset(set(cols)):
                        labels_to_use = preferred
                    else:
                        labels_to_use = cols
            except Exception:
                labels_to_use = None
            if labels_to_use is None:
                # Fall back to observed labels; if they match the preferred set/subset, use preferred.
                observed = sorted(pd.unique(pd.concat([y_true.astype(str), y_pred.astype(str)]).astype(str)))
                if set(observed).issubset(set(preferred)) or set(preferred).issubset(set(observed)):
                    labels_to_use = preferred
                else:
                    labels_to_use = observed
        cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels_to_use)
        pd.DataFrame(cm).to_csv(os.path.join(output_dir, f'{split_name}_confusion_matrix.csv'), index=False, header=False)
        # Save for console printing later (under feature importance output)
        printed_artifacts[split_name] = {
            'classification_report': rep,
            'confusion_matrix': cm,
            'labels': labels_to_use,
        }
        # Predictions + probabilities
        preds_path = os.path.join(output_dir, f'{split_name}_predictions.csv')
        out_pred = split_df[[label]].copy()
        out_pred['pred'] = y_pred
        try:
            proba_df = predictor.predict_proba(X)
            if problem_type == 'multiclass' and isinstance(proba_df, pd.DataFrame):
                conf_top1 = proba_df.max(axis=1)
                y_true_str = y_true.astype(str)
                col_idx = proba_df.columns.get_indexer(y_true_str)
                p_true = pd.Series(proba_df.to_numpy()[np.arange(len(proba_df)), col_idx], index=proba_df.index)

                # Attach proba columns and minimal diagnostics
                proba_renamed = proba_df.rename(columns={c: f'p_{str(c)}' for c in proba_df.columns})
                out_pred = pd.concat([
                    out_pred,
                    proba_renamed,
                    pd.DataFrame({
                        'conf_top1': conf_top1,
                        'p_true': p_true,
                    })
                ], axis=1)

                # Aggregated uncertainty metrics
                # NLL (negative log-likelihood)
                nll = float((-np.log(p_true.clip(lower=1e-12))).mean())
                # Brier score (multiclass)
                p_sq_sum = (proba_df ** 2).sum(axis=1)
                brier = float((p_sq_sum - 2 * p_true + 1.0).mean())
                # ECE (top-1, 10 bins)
                bins = np.linspace(0.0, 1.0, 11)
                conf = conf_top1
                correct = (y_pred == y_true)
                ece = 0.0
                N = len(conf)
                for b_start, b_end in zip(bins[:-1], bins[1:]):
                    mask = (conf >= b_start) & (conf < b_end) if b_end < 1.0 else (conf >= b_start) & (conf <= b_end)
                    if mask.sum() == 0:
                        continue
                    bin_acc = correct[mask].mean()
                    bin_conf = conf[mask].mean()
                    ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
                metrics[f'{split_name}_nll'] = nll
                metrics[f'{split_name}_brier'] = brier
                metrics[f'{split_name}_ece_top1_10bins'] = float(ece)
            else:
                out_pred['p_positive'] = proba_df
        except Exception:
            pass
        out_pred.to_csv(preds_path)

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # Feature importance
    eval_df = val_df if len(val_df) else train_df
    imp_df = compute_feature_importance_ag(
        predictor,
        eval_df,
        features,
        label,
        subsample_size=min(importance_subsample, len(eval_df)),
        time_limit=importance_time_limit,
    )
    os.makedirs(reports_fig_dir, exist_ok=True)
    imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    fi_png = os.path.join(reports_fig_dir, 'feature_importance.png')
    plot_feature_importance(imp_df, top_n=importance_top_n, output_path=fi_png)
    # Console preview (compact)
    top_n = int(min(15, max(1, importance_top_n)))
    top = imp_df.sort_values('importance', ascending=False).head(top_n)
    if not top.empty and {'feature', 'importance'}.issubset(top.columns):
        print('Top feature importances:')
        print(top[['feature', 'importance']].to_string(index=False))

    return metrics