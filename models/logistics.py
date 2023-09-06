import utils.features as ft
from sklearn.linear_model import LogisticRegression
import numpy as np


def flatten_sequences(array):
    # USE: flatten the last two dims of the array produced by create_sequences()
    # INPUT: array
    # OUTPUT: array

    shape = array.shape
    array = array.reshape(shape[0], shape[1] * shape[2])

    return array


def train_pred(df,
               features, target, lags, forward,
               target_in_forward,
               val_percent, test_percent,
               pos_weight,
               random_seed,
               ):
    # USE: use logistics regression to pred water level surge
    # INPUT: df, pandas df
    #        target_col, name of target col in df
    #        pred_forward, number of time steps to predict
    #        test_index, list, the index of test targets
    #        test_df, incomplete test df
    # OUTPUT: df, one col is true, another is pred

    # datasets
    sequences = ft.create_sequences(df, lags, forward, features, target)
    train_x, train_y, val_x, val_y, test_x, test_y = ft.split_sequences(
        sequences,
        val_percent, test_percent,
        forward,
        random_seed,
        shuffle=False)

    train_x = np.concatenate([train_x, val_x], axis=0)
    train_y = np.concatenate([train_y, val_y], axis=0)

    train_x = flatten_sequences(train_x)
    train_y = flatten_sequences(train_y)[:, target_in_forward - 1]
    test_x = flatten_sequences(test_x)
    test_y = flatten_sequences(test_y)[:, target_in_forward - 1]

    # train
    model = LogisticRegression(
        class_weight=pos_weight,
        max_iter=150,
    ).fit(train_x, train_y)

    # pred
    pred_y = model.predict(test_x)

    # store results
    val_count = int(np.floor(sequences.shape[0] * val_percent))
    test_count = int(np.floor(sequences.shape[0] * test_percent))
    train_count = sequences.shape[0] - val_count - test_count
    test_df = df.iloc[train_count + val_count: train_count + val_count + test_count]
    test_df = test_df[[target]].rename(columns = {target: 'true'})
    test_df['pred'] = pred_y

    return test_df
