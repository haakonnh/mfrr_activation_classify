###############################################################################
# utility functions for data processing
###############################################################################


import pandas as pd


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample dataframe to 15-min frequency using forward fill."""
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('15min').ffill()
    return df