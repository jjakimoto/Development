import json

import pandas as pd

from libs.model import Model
from libs.features import load_feature_config, subsample, calc_return, generate_features
from libs.pandas_utils import get_non_duplicated, subsample_returns


def generate_signal_prices(dfs, feature_names_filename, model_params_filename,
                           period=1e7, sample_th=10, returns_th=5):
    file = open(model_params_filename, 'r')
    model_params = json.load(file)
    file.close()
    features_config = load_feature_config(feature_names_filename)
    timestamps = [subsample(df, sample_th) for df in dfs]
    model = Model(**model_params)
    returns = [calc_return(df, period) for df in dfs]
    features = [generate_features(df, features_config) for df in dfs]
    returns = [calc_return(df, period) for df in dfs]
    returns = [get_non_duplicated(x) for x in returns]
    features = [get_non_duplicated(feature) for feature in features]
    sub_features = [features[i].loc[timestamps[i]] for i in range(len(timestamps))]
    sub_returns = [returns[i].loc[timestamps[i]] for i in range(len(timestamps))]
    signal_prices_list = []
    for i in range(len(features)):
        test_features = features[i]
        X_test = get_non_duplicated(test_features)
        cond = (X_test.isnull().sum(axis=1) == 0)
        X_test = X_test.loc[cond]
        df = get_non_duplicated(dfs[i])
        df = df.loc[cond]
        mid_prices = 0.5 * (df['ap0'] + df['bp0'])
        X_train = []
        y_train = []
        for j in range(len(features)):
            if i == j:
                continue
            train_features = sub_features[j]
            train_returns = sub_returns[j]
            cond = (~train_returns.isnull()) & (train_features.isnull().sum(axis=1) == 0)
            train_features = train_features.loc[cond]
            train_returns = train_returns.loc[cond]
            X_train.append(train_features)
            y_train.append(train_returns)
        X_train = pd.concat(X_train)
        y_train = pd.concat(y_train)
        X_train, y_train = subsample_returns(X_train, y_train, returns_th)
        model.fit(X_train.values, y_train.values)
        pred = model.predict(X_test)
        pred = pd.Series(pred, index=X_test.index)
        signal_prices = mid_prices + pred
        signal_prices_list.append(signal_prices)
    return signal_prices_list


if __name__ == '__main__':
    # Load Data
    dfs = []
    for i in range(5):
        df = pd.read_csv(f'outputs/output_res_2019061{i}.csv', index_col='timestamp')
        dfs.append(df)
    period = 1e7
    sample_th = 10
    returns_th = 5
    feature_names_filename = "feature_names.txt"
    model_params_filename = "model_params.txt"
    fair_prices = generate_signal_prices(dfs, feature_names_filename,
                                         model_params_filename, period,
                                         sample_th, returns_th)
    # Save results
    for i in range(len(fair_prices)):
        filename = f'fair_prices/fair_prices{i}.csv'
        fair_price = fair_prices[i]
        fair_price.to_csv(filename, index=True, header=True)
