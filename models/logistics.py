import utils.features as ft
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def flatten_sequences(array):
    # USE: flatten the last two dims of the array produced by create_sequences()
    # INPUT: array
    # OUTPUT: array

    shape = array.shape
    if len(shape) > 2:
        array = array.reshape(shape[0], shape[1] * shape[2])

    return array


def train_pred(df,
               features, target, lags, forward,
               target_in_forward,
               val_percent, test_percent,
               if_weight,
               random_seed,
               ):
    # USE: use logistics regression to pred water level surge
    # INPUT: df, pandas df
    #        features, list of str, col names in df
    #        target, str, col names in df
    #        lags, forward, list of ints, starting from 1
    #        target_in_forward, the position of target in forward list, starting from 1
    #        val_percent, test_percent, float, 0-1
    #        pos_weight, the weight of targets used for the scikit logistics regression
    # OUTPUT: df, one col is true, another is pred

    # datasets
    df['index'] = range(len(df))
    sequences_w_index = ft.create_sequences(df, lags, forward, features + ['index'], target)
    train_x, train_y, val_x, val_y, test_x, test_y = ft.split_sequences(
        sequences_w_index[:, :, :-1],
        val_percent, test_percent,
        forward,
        random_seed,
        shuffle=False)
    _, _, _, _, _, test_y_w_index = ft.split_sequences(
        sequences_w_index[:, :, [-1]], val_percent, test_percent, forward, random_seed, shuffle=False)

    train_x = np.concatenate([train_x, val_x], axis=0)
    train_y = np.concatenate([train_y, val_y], axis=0)

    train_x = flatten_sequences(train_x)
    train_y = flatten_sequences(train_y)[:, target_in_forward - 1]
    test_x = flatten_sequences(test_x)
    test_y = flatten_sequences(test_y)[:, target_in_forward - 1]

    if if_weight:
        num_class = df[target].nunique()
        weights = len(df) / (num_class * np.bincount(df[target].values.astype(int)))
        weight_dict = {class_name: weight for class_name, weight in zip(np.arange(num_class), weights)}
    else:
        weight_dict = None

    # train
    model = LogisticRegression(
        class_weight=weight_dict,
        max_iter=150,
    ).fit(train_x, train_y)

    # pred
    pred_y = model.predict(test_x)

    # store results
    test_y_datatime_index = df.iloc[test_y_w_index[:, target_in_forward - 1]].index
    test_df = pd.DataFrame(test_y, columns=['true'], index=test_y_datatime_index)
    test_df['pred'] = pred_y

    return test_df
