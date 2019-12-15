import json
from collections import defaultdict
# Thrid party
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
from sklearn.metrics import log_loss
# Custom library
from .stats import EMA, SUM
from .pandas_utils import get_non_duplicated


# Colection of Features
#################################################
def calc_order_flow_imbalacne(df, period, verbose=0):
    """ Calulate summation of order flow at BBO with specified period"""
    ofi_sum = SUM(period)
    ofis = []
    indices = range(df.shape[0])
    if verbose == 1:
        indices = tqdm_notebook(indices)
    elif verbose == 2:
        indices = tqdm(indices)
    for i in indices:
        x = df.iloc[i]
        timestamp = df.index[i]
        ofi_sum.update(x['ofi'], timestamp)
        ofi = ofi_sum.get_signal()
        ofis.append(ofi)
    ofis = pd.Series(ofis, index=df.index)
    # insert nan for unstable period
    ofis.loc[:period] = np.nan
    return ofis


def smooth_signals(signals, period, verbose=0):
    """ Smooth out signals with EMA """
    ema = EMA(period)
    _smooth_output = []
    indices = range(signals.shape[0])
    if verbose == 1:
        indices = tqdm_notebook(indices)
    elif verbose == 2:
        indices = tqdm(indices)
    for i in indices:
        x = signals.iloc[i]
        if np.isfinite(x):
            timestamp = signals.index[i]
            ema.update(x, timestamp)
            smooth_signal = ema.get_signal()
        else:
            smooth_signal = np.nan
        _smooth_output.append(smooth_signal)
    smooth_output = pd.Series(_smooth_output, index=signals.index)
    # insert nan for unstable period
    smooth_output.loc[:period] = np.nan
    return smooth_output


def calc_smooth_stats(prices, period, verbose=0):
    """ Calculate smoothed price statistics"""
    mean_ema = EMA(period)
    vol_ema = EMA(period)
    vols = []
    mean_prices = []
    indices = range(prices.shape[0])
    if verbose == 1:
        indices = tqdm_notebook(indices)
    elif verbose == 2:
        indices = tqdm(indices)
    for i in indices:
        price = prices.iloc[i]
        if np.isfinite(price):
            timestamp = prices.index[i]
            mean_ema.update(price, timestamp)
            mean_price = mean_ema.get_signal()
            vol_ema.update(abs(price - mean_price), timestamp)
            vol = vol_ema.get_signal()
        else:
            vol = np.nan
            mean_price = np.nan
        vols.append(vol)
        mean_prices.append(mean_price)
    vols = pd.Series(vols, index=prices.index)
    mean_prices = pd.Series(mean_prices, index=prices.index)
    zscores = (prices - mean_prices) / vols
    # insert nan for unstable period
    mean_prices.loc[:period] = np.nan
    vols.loc[:period] = np.nan
    zscores.loc[:period] = np.nan
    return mean_prices, vols, zscores


def calc_micro_price(df, level=1, verbose=0):
    """ Calculate fair price based on volume imblance"""
    prices = []
    indices = range(df.shape[0])
    if verbose == 1:
        indices = tqdm_notebook(indices)
    elif verbose == 2:
        indices = tqdm(indices)
    for i in indices:
        x = df.iloc[i]
        av = 0
        bv = 0
        ap = x['ap0']
        bp = x['bp0']
        if np.isfinite(ap) and np.isfinite(bp):
            for i in range(level):
                av += x[f'aq{i}']
                bv += x[f'bq{i}']
            price = (ap * bv + bp *av) / (av + bv)
        else:
            price = np.nan
        prices.append(price)
    prices = pd.Series(prices, index=df.index)
    return prices


def calc_return(df, period):
    """ Calculate mid price return"""
    prices = 0.5 * (df['ap0'] + df['bp0'])
    forward_index = prices.index.searchsorted(prices.index + period)
    forward_index[forward_index == prices.shape[0]] = prices.shape[0] - 1
    ret = prices.values[forward_index] - prices
    ret.loc[prices.index[-1] - period:] = np.nan
    return ret


# Feature generator based on configuration
##########################################
def generate_single_feature(df, name, period):
    """ Generate feature with name and period

    Params
    ------
    df: pd.DataFrame
    name: str
        The name of feature. That has to be either of 'vol', 'zscore', 'ofi',
        or 'volume_imbalance'.
    period: float
        Smoothing or summing period
    
    Returns
    -------
    pd.Series
    """
    if name in ["vol", 'zscore']:
        mid_prices = 0.5 * (df['ap0'] + df['bp0'])
        mean_prices, vols, zscores = calc_smooth_stats(mid_prices, period)
        if name == 'vol':
            return vols
        else:
            return zscores
    elif name == 'ofi':
        ofi = calc_order_flow_imbalacne(df, period)
        return ofi
    elif name == 'volume_imbalance':
        volume_imb = df['bq0'] - df['aq0']
        smooth_volume_imb = smooth_signals(volume_imb, period)
        return smooth_volume_imb
    else:
        raise NotImplementedError(f"No Implementation name={name}")

     
def generate_features(df, features_config):
    """ Generate feature with multiple names and periods

    Params
    ------
    df: pd.DataFrame
    features_config: dict(list)
        Key is name of feature and value is list of periods
    
    Returns
    -------
    pd.DataFrame
    """
    features = dict()
    for name, periods in features_config.items():
        for period in periods:
            _name = f'{name}_{period}'
            features[_name] = generate_single_feature(df, name, period)
    return pd.DataFrame(features)


# Parameter Selection
#########################################
def feat_imp_MDA(clf, features, labels):
    """Calculate Mean Decrease Accuracy
    
    Params
    ------
    clf: Classifier instance
    features: list(pd.DataFrae)
    labels:  list(pd.Series) 
    
    Returns
    -------
    imp: pd.DataFrame, feature importance of means and standard deviations
    scores: float, scores of cross validation
    """
    index = np.arange(len(features))
    scores = pd.Series(index=index)
    scores_perm = pd.DataFrame(index=index, columns=features[0].columns)
    for i in range(len(features)):
        test_df = features[i]
        test_label = labels[i]
        X_test = test_df.loc[~test_df.index.duplicated()]
        y_test = test_label.loc[~test_label.index.duplicated()]
        # Consider non zero label case
        cond = (~y_test.isnull()) & (X_test.isnull().sum(axis=1) == 0) & (y_test != 0)
        X_test = X_test.loc[cond]
        y_test = y_test.loc[cond]
        X_train = []
        y_train = []
        for j in range(len(features)):
            if i == j:
                continue
            df = features[j]
            label = labels[j]
            df = df.loc[~df.index.duplicated()]
            label = label.loc[~label.index.duplicated()]
            cond = (~label.isnull()) & (df.isnull().sum(axis=1) == 0) & (label != 0)
            df = df.loc[cond]
            label = label.loc[cond]
            X_train.append(df)
            y_train.append(label)
        X_train = pd.concat(X_train)
        y_train = pd.concat(y_train)
        clf_fit = clf.fit(X_train.values, y_train.values)
        proba = clf_fit.predict_proba(X_test)[:, 1]
        score = -log_loss(y_test, proba)
        scores.loc[i] = score

        for col in X_test.columns:
            X_test_ = X_test.copy(deep=True)
            # Randomize certain feature to make it not effective
            np.random.shuffle(X_test_[col].values)
            proba = clf_fit.predict_proba(X_test_.values)[:, 1]
            score = -log_loss(y_test.values, proba)
            scores_perm.loc[i, col] = score
    # (Original score) - (premutated score)
    imprv = (-scores_perm).add(scores, axis=0)
    # Relative to maximum improvement
    max_imprv = -scores_perm
    imp = imprv / max_imprv
    imp = pd.DataFrame(
        {'mean': imp.mean(), 'std': imp.std() * np.sqrt(imp.shape[0])})
    return imp, scores.mean()


# Subsampling
##############################
def subsample(df, sample_th):
    """ sample timestamps based on order flow at BBO

    Params
    ------
    df: pd.DataFrame
    sample_th: int
        Sampling threashold. For every time cumlative value exceeds
        this value, we sample timestamp
    
    Returns
    -------
    list(int), sampled timestamp
    """
    cum_ofi = np.abs(df['ofi']).cumsum()
    ofi = get_non_duplicated(cum_ofi).diff().dropna()
    cum_val = 0
    timestamps = []
    for i, x in enumerate(ofi.values):
        cum_val += abs(x)
        if cum_val >= sample_th:
            timestamps.append(ofi.index[i])
            cum_val = 0
    return timestamps


def load_feature_config(filename):
    """ Load feature from file"""
    file = open(filename, 'r')
    select_names = json.load(file)
    file.close()
    features_config = defaultdict(list)
    for key in select_names:
        key = key.split('_')
        period = float(key[-1])
        name = "_".join(key[:-1])
        features_config[name].append(period)
    features_config = dict(features_config)
    return features_config
