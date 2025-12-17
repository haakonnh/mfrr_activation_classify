# Preprocessing cache

Goal: run feature engineering once, reuse the same dataframe for many training runs.

## 1) Build the cache

```bat
.venv1\Scripts\python.exe upreg_classify\scripts\preprocess_cache.py --area NO1 --out upreg_classify\data\preprocessed\df_NO1.pkl
```

To overwrite an existing cache:

```bat
.venv1\Scripts\python.exe upreg_classify\scripts\preprocess_cache.py --area NO1 --out upreg_classify\data\preprocessed\df_NO1.pkl --force
```

## 2) Train using the cache

```bat
.venv1\Scripts\python.exe upreg_classify\src\train\train.py --task multiclass --preprocessed_path upreg_classify\data\preprocessed\df_NO1.pkl
```

To force recomputing preprocessing (ignores any existing cache file):

```bat
.venv1\Scripts\python.exe upreg_classify\src\train\train.py --task multiclass --preprocessed_path upreg_classify\data\preprocessed\df_NO1.pkl --recompute_preprocess
```
