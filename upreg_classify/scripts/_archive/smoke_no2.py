import os, sys, importlib.util
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # upreg_classify
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)
PREPROCESS_PATH = os.path.join(PKG_ROOT, 'src', 'data', 'preprocess.py')
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_PATH)
preprocess = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys as _sys
_sys.modules[spec.name] = preprocess
spec.loader.exec_module(preprocess)
Config = preprocess.Config
build_dataset = preprocess.build_dataset

cfg = Config(
    data_dir=os.path.join('upreg_classify','data','raw'),
    area='NO2',
    include_2024=True,
    dropna=True,
    heavy_interactions=False,
    train_frac=0.02,
    val_frac=0.01,
    test_frac=0.0,
)

df, (train_df, val_df, test_df), features = build_dataset(cfg, label_name='RegClass+4')
print('DF shape:', df.shape)
print('Splits:', len(train_df), len(val_df), len(test_df))
print('Features:', len(features))
print('Columns sample:', features[:10])
