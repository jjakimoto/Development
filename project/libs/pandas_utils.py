import numpy as np
import pandas as pd

def get_non_duplicated(df):
    return df.loc[~df.index.duplicated(keep='last')]


def subsample_returns(features, returns, return_th, min_subsample=1e-2):
    """ Subsample small return sample"""
    # Add hoc down sampling to balance zero and non zero return
    cond = np.abs(returns) >= return_th
    returns_ = returns.loc[cond]
    features_ = features.loc[cond]
    cond = np.abs(returns) < return_th
    returns0 = returns.loc[cond]
    features0 = features.loc[cond]
    # Maximum subsampling is 1/100
    returns_size = max(len(returns_), min_subsample * len(returns0))
    sample_freq = int(max(len(returns0) / returns_size, 1))
    features0 = features0.iloc[::sample_freq]
    returns0 = returns0.iloc[::sample_freq]
    features = pd.concat((features_, features0))
    returns = pd.concat((returns_, returns0))
    return features, returns