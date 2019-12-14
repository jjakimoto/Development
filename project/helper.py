# Default libralies
import itertools
import time
from datetime import datetime
import sys
from copy import deepcopy
import multiprocessing as mp
import multiprocessing.pool
import functools
# Thrid party
import math
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
from sklearn.metrics import log_loss
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class SUM:
    def __init__(self, period):
        self._period = period
        self._value = 0
        self._timestamps = list()
        self._values = list()
        
    def update(self, value, timestamp):
        while len(self._timestamps) > 0 and timestamp - self._timestamps[0] > self._period:
            self._value -= self._values[0]
            self._timestamps.pop(0)
            self._values.pop(0)
        self._value += value
        self._values.append(value)
        self._timestamps.append(timestamp)
            
    def get_signal(self):
        return self._value



class EMA:
    def __init__(self, period):
        self._tau = 1. / period
        self._value = 0
        self._timestamp = 0
        self._start = False
        
    def update(self, value, timestamp):
        if self._start:
            delta = timestamp - self._timestamp
            alpha = math.exp(-delta  * self._tau)
            self._value = (1 - alpha) * value + alpha * self._value
        else:
            self._value = value
            self._start = True
        self._timestamp = timestamp
            
    def get_signal(self):
        return self._value


def grid_samples(params_conf, condition=None):
    """Grid Sampling

    Args:
        params_conf (list(dict) or dict): sample configuration

    Returns:
        samples (list(dict))
    """
    domains = list()
    names = []
    for conf in params_conf:
        names.append(conf['name'])
        domain = conf["domain"]
        type_ = conf.get("type", "grid")
        if type_ in ["integer", "continuous"]:
            if conf.get("scale", None) == 'log':
                domain = np.logspace(np.log10(domain[0]), np.log10(domain[1]), conf["num_grid"])
            else:
                domain = np.linspace(domain[0], domain[1], conf["num_grid"])
            if type_ == "integer":
                domain = domain.astype(int)
        elif type_ == "fixed":
            domain = [domain, ]
        domains.append(list(domain))

    patterns = itertools.product(*domains)
    samples = list()
    for params_val in patterns:
        params_dict = dict()
        for name, val in zip(names, params_val):
            params_dict[name] = val
        if condition is None or condition(params_dict):
            samples.append(params_dict)
    return samples


def feat_imp_MDA(clf, X, y, sample_weight=None, scoring='neg_log_loss', n_splits=3, t1=None,
                 cv_gen=None, pct_embargo=0, purging=True, num_threads=1):
    """Calculate Mean Decrease Accuracy
    
    Params
    ------
    clf: Classifier instance
    X: pd.DataFrame, Input feature
    y: pd.Series, Label        
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    scoring: str, default 'neg_log_loss'
        The name of scoring methods. 'f1', 'accuracy' or 'neg_log_loss'
    n_splits: int, default 3
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    cv_gen: KFold instance
        If not specified, use PurgedKfold
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for purging
    
    Returns
    -------
    imp: pd.DataFrame, feature importance of means and standard deviations
    scores: float, scores of cross validation
    """
    
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo,
                             purging=purging, num_threads=num_threads)
    index = np.arange(n_splits)
    scores = pd.Series(index=index)
    scores_perm = pd.DataFrame(index=index, columns=X.columns)
    for idx, (train, test) in zip(index, cv_gen.split(X=X)):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        if sample_weight is not None:
            w_train = sample_weight.iloc[train].values
        else:
            w_train = None
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        if sample_weight is not None:
            w_test = sample_weight.iloc[test].values
        else:
            w_test = None
        clf_fit = clf.fit(X_train, y_train, sample_weight=w_train)
        scores.loc[idx] = evaluate(clf_fit, X_test, y_test, scoring,
                                   sample_weight=w_test, labels=clf_fit.classes_)

        for col in X.columns:
            X_test_ = X_test.copy(deep=True)
            # Randomize certain feature to make it not effective
            np.random.shuffle(X_test_[col].values)
            scores_perm.loc[idx, col] = evaluate(clf_fit, X_test_, y_test, scoring,
                                                 sample_weight=w_test, labels=clf_fit.classes_)
    # (Original score) - (premutated score)
    imprv = (-scores_perm).add(scores, axis=0)
    # Relative to maximum improvement
    if scoring == 'neg_log_loss':
        max_imprv = -scores_perm
    else:
        max_imprv = 1. - scores_perm
    imp = imprv / max_imprv
    imp = pd.DataFrame(
        {'mean': imp.mean(), 'std': imp.std() * np.sqrt(imp.shape[0])})
    return imp, scores.mean()


def expand_call(kwargs):
    """Execute function from dictionary input"""
    func = kwargs['func']
    del kwargs['func']
    out = func(**kwargs)
    return out


def report_progress(job_idx, num_jobs, time0, task):
    """Report progress to system output"""
    msg = [float(job_idx) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(datetime.fromtimestamp(time.time()))
    msg_ = time_stamp + ' ' + str(
        round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
           str(round(msg[1], 2)) + ' minutes. Remaining ' + str(
        round(msg[2], 2)) + ' minutes.'
    if job_idx < num_jobs:
        sys.stderr.write(msg_ + '\r')
    else:
        sys.stderr.write(msg_ + '\n')


def process_jobs(jobs, task=None, num_threads=24, use_thread=False):
    """Execute parallelized jobs

    Parameters
    ----------
    jobs: list(dict)
        Each element contains `function` and its parameters
    task: str, optional
        The name of task. If not specified, function name is used
    num_threads, int, optional
        The number of threads for parallelization. if not feeded, use the maximum
        number of process
    use_thread: bool, defulat False
        If True, use multi process. If False, use multi thread.
        Use True, if the multiprocessing exceeds the memory limit

    Returns
    -------
    List: each element is results of each part
    """
    if task is None:
        task = jobs[0]['func'].__name__
    if num_threads is None:
        num_threads = mp.cpu_count()
    out = []
    if num_threads > 1:
        if use_thread:
            pool = mp.pool.ThreadPool(processes=num_threads)
        else:
            pool = mp.Pool(processes=num_threads)
        outputs = pool.imap_unordered(expand_call, jobs)
        time0 = time.time()
        # Execute programs here
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            report_progress(i, len(jobs), time0, task)
        pool.close()
        pool.join()
    else:
        for job in jobs:
            job = deepcopy(job)
            func = job['func']
            del job['func']
            out_ = func(**job)
            out.append(out_)
    return out


def multi_grid_execution(params_conf, task_func, default_params=None,
                         condition=None, num_threads=None):
    samples = grid_samples(params_conf, condition=condition)
    if default_params is None:
        default_params = dict()
    jobs = []
    for sample in samples:
        sample['func'] = functools.partial(task_func,**default_params)
        jobs.append(sample)
    output = process_jobs(jobs, num_threads=num_threads, task='task')
    return output


# Feature related functions
####################################################
def calc_order_flow_imbalacne(df, period, verbose=0):
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
    prices = 0.5 * (df['ap0'] + df['bp0'])
    forward_index = prices.index.searchsorted(prices.index + period)
    forward_index[forward_index == prices.shape[0]] = prices.shape[0] - 1
    ret = prices.values[forward_index] - prices
    ret.loc[prices.index[-1] - period:] = np.nan
    return ret


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


# Functions for generating signals
##################################################
def generate_single_feature(df, name, period):
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
    features = dict()
    for name, periods in features_config.items():
        for period in periods:
            _name = f'{name}_{period}'
            features[_name] = generate_single_feature(df, name, period)
    return pd.DataFrame(features)


class Model:
    def __init__(self, **kwargs):
        self._model = Lasso(**kwargs)
        self._pre = StandardScaler(with_mean=False)
        
    def fit(self, X, y):
        X = self._pre.fit_transform(X)
        self._model.fit(X, y)
        
    def predict_clf(self, X, th):
        pred = self.predict(X)
        output = np.zeros_like(pred)
        output[pred >= th] = 1
        output[pred <= -th] = -1
        return output
    
    def predict(self, X):
        X = self._pre.transform(X)
        pred = self._model.predict(X)
        return pred