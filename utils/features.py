import pandas as pd
import numpy as np
from pandas.api.types import is_list_like
from typing import List, Tuple
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import math
import os

import torch

import utils.preprocess as pp


def _get_32_bit_dtype(x):
    dtype = x.dtype
    if dtype.name.startswith("float"):
        redn_dtype = "float32"
    elif dtype.name.startswith("int"):
        redn_dtype = "int32"
    else:
        redn_dtype = None
    return redn_dtype


def add_lags(
        df: pd.DataFrame,
        lags: List[int],
        column: str,
        ts_id: str = None,
        use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe
    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(lags), "`lags` should be a list of all required lags"
    assert (
            column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        pass
        # warnings.warn(
        #     "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        # )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df[column].shift(l).astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (
                ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column]
                .shift(l)
                .astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
            }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features


def update_lag_w_forward(lag, forward):
    # USE: update lag list using forward list
    #      new lag list is respect to the furthest time step in the forward list
    #      the basic idea of the conversion is moving the "0" tick to before each of the forward item
    # INPUT: both are lists
    # OUTPUT: updated lag list

    count = 0
    for f in forward:
        lag = [l + (f - 1) for l in lag]
        forward = [ff - (f - 1) for ff in forward]
        count += 1
        if count > 1:
            lag = [1] + lag

    return lag


def create_sequences(df, lag, forward, feature_list, target=[]):
    # USE: create sequences with lags for feature and target cols
    # INPUT: df: pandas df
    #        lag: list, starting from 1
    #        feature_list: list of feature names in df
    #        target_list: list of target names in df

    # convert lag and forward to the lag list of furthest target
    lag_update = update_lag_w_forward(lag, forward)

    sequences = np.empty((len(df) - ((lag_update[-1]) + 1) + 1, len(lag_update) + 1, 1))
    for v in (target + feature_list):
        df_lag = add_lags(df, lags=lag_update, column=v)[0]
        # df_lag = df_lag.dropna()
        df_lag = df_lag[len(lag):]

        # make up for the missing rows to keep rows consecutive
        minute_delta = df_lag.index.to_series().diff().mode()[0].total_seconds() / 60
        full_range = pd.date_range(start=df_lag.index.min(), end=df_lag.index.max(), freq=f'{int(minute_delta)}T')
        df_lag = df_lag.reindex(full_range)

        drop_list = feature_list + target
        drop_list.remove(v)
        df_lag = df_lag.drop(columns=drop_list, axis=1)

        df_lag = df_lag[df_lag.columns[::-1]]

        sequences = np.concatenate((sequences, df_lag.values[:, :, np.newaxis]), axis=-1)

    sequences = sequences[:, :, 1:]
    return sequences


def _add_time(x, num_time_feature, start_time):
    len_x = x.shape[1]
    for num in range(num_time_feature):
        num_array = np.arange(len_x) + start_time
        num_array_expand = np.expand_dims(np.tile(num_array, [x.shape[0], 1]), axis=2)
        x = np.concatenate((x, num_array_expand), axis=2)
        start_time += 1
    return x


def _addLoc(x, numLocFeature, startLoc):
    len_X = x.shape[1]
    for num in range(numLocFeature):
        num_array = np.zeros(len_X) + startLoc
        num_array_expand = np.expand_dims(np.tile(num_array, [x.shape[0], 1]), axis=2)
        x = np.concatenate((x, num_array_expand), axis=2)
        startLoc += 1
    return x


def split_sequences(
        sequences,
        train_percent, val_percent, test_percent, forward,
        num_time_feature=0, num_loc_feature=0
):
    # USE: split the sequences into datasets
    # INPUT: sequences: array, three dims (sample, lag, feature)
    #        val_proportion, test_proportion: float, 0-1
    #        forward: list, time steps forward to predict
    #        num_time_feature, num_loc_feature:
    #        shuffle: bool, if shuffle the sequences
    #        random_seed: int
    # OUTPUT: datasets: array

    val_count = int(np.floor(sequences.shape[0] * val_percent))
    test_count = int(np.floor(sequences.shape[0] * test_percent))
    train_count = int(np.floor(sequences.shape[0] * train_percent))

    forward_count = len(forward)

    # indexing from the first time step
    train_x = sequences[:train_count, :-forward_count, :]
    train_y = sequences[:train_count, -forward_count:, 0]
    val_x = sequences[train_count: train_count + val_count, :-forward_count, :]
    val_y = sequences[train_count: train_count + val_count, -forward_count:, 0]
    test_x = sequences[train_count + val_count: train_count + val_count + test_count, :-forward_count, :]
    test_y = sequences[train_count + val_count: train_count + val_count + test_count, -forward_count:, 0]

    if num_time_feature > 0:
        train_x = _add_time(train_x, num_time_feature, 1)
        val_x = _add_time(val_x, num_time_feature, 1)
        test_x = _add_time(test_x, num_time_feature, 1)

    if num_loc_feature > 0:
        train_x = _addLoc(train_x, num_loc_feature, 1)
        val_x = _addLoc(val_x, num_loc_feature, 1)
        test_x = _addLoc(test_x, num_loc_feature, 1)

    print('Dims: (sample, sequenceLen, feature)')
    print('trainX shape:', train_x.shape, ' trainY shape:', train_y.shape)
    print('valX shape:', val_x.shape, ' valY shape:', val_y.shape)
    print('testX shape:', test_x.shape, ' testY shape:', test_y.shape)
    return train_x, train_y, val_x, val_y, test_x, test_y


def create_index_4_cv(x, if_cv, num_fold_cv, val_percent, test_percent, val_percent_cv, test_percent_cv):
    dataset_index = []
    # get the index at the sample dim without cv
    if not if_cv:
        # num_fold = 2
        # tscv = TimeSeriesSplit(num_fold, test_size=round(x.shape[0] * test_percent))
        # for i, (train_index, test_index) in enumerate(tscv.split(x)):
        #     test_size = val_percent / (1 - (num_fold - i) * test_percent)
        #     test_size = 0.9 if test_size > 1 else test_size  # to avoid too much test data error
        #     train_index, val_index, _, _ = train_test_split(train_index, train_index,
        #                                                     test_size=test_size,
        #                                                     shuffle=False)
        #     # print(f"Fold {i}:")
        #     # print(f"  Train: index={train_index}")
        #     # print(f"  Val: index={val_index}")
        #     # print(f"  Test:  index={test_index}")
        #     dataset_index.append({'train_index': train_index, 'val_index': val_index, 'test_index': test_index})
        # dataset_index = dataset_index[-1:]
        # # print("Only keep the fold at the end of the time series.")
        test_index = list(range(x.shape[0]))[-round(x.shape[0] * test_percent):]
        train_index = list(range(x.shape[0]))[: -round(x.shape[0] * test_percent)]
        train_index, val_index, _, _ = train_test_split(train_index, train_index,
                                                        test_size=val_percent / (1 - test_percent),
                                                        shuffle=False)
        dataset_index.append({'train_index': train_index, 'val_index': val_index, 'test_index': test_index})

    # get the index at the sample dim with cv
    else:
        tscv = TimeSeriesSplit(num_fold_cv, test_size=math.floor(x.shape[0] * test_percent_cv))
        for i, (train_index, test_index) in enumerate(tscv.split(x)):
            train_index, val_index, _, _ = train_test_split(train_index, train_index,
                                                            test_size=val_percent_cv / (
                                                                    1 - (num_fold_cv - i) * test_percent_cv),
                                                            shuffle=False)
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Val: index={val_index}")
            print(f"  Test:  index={test_index}")
            dataset_index.append({'train_index': train_index, 'val_index': val_index, 'test_index': test_index})

    return dataset_index


def update_test_percent(df_field, df_normed, sequences_w_index, test_percent):

    # get the start time of test samples
    num_test_sample = math.ceil(len(df_field) * test_percent)
    time_first_test_sample = df_field.iloc[-num_test_sample].name

    # convert from timestamp to sample index
    test_sample_index = df_normed[df_normed.index >=
                                  (time_first_test_sample - pd.to_timedelta(1, unit='h'))]['index'].values

    # get the start index of test samples
    num_test_sequences = np.sum(sequences_w_index[:, -1, -1] >= test_sample_index[0])
    test_percent_updated = num_test_sequences / sequences_w_index.shape[0]
    # test_percent_updated = test_percent_updated + 0.01  # for potential rounding issue
    return test_percent_updated, df_field.iloc[-num_test_sample:], num_test_sequences


def split_df_field(df_field, test_df_field, val_percent, test_percent_updated):
    train_df_field = df_field[~df_field.index.isin(test_df_field.index)]

    train_df_field_sorted = train_df_field.sort_values(by='discharge')
    train_df_field_sorted['bin'] = pd.cut(train_df_field_sorted['discharge'], 10, duplicates='drop')
    val_percent_updated = val_percent / (1 - test_percent_updated)
    val_df_field = train_df_field_sorted.groupby('bin', observed=False).apply(
        lambda x: x.sample(math.floor(len(x) * val_percent_updated), random_state=0)).droplevel(0)
    train_df_field = train_df_field[~train_df_field.index.isin(val_df_field.index)]
    return train_df_field, val_df_field


def filter_df_field(train_df_field, val_df_field, df_normed, sample_indice):

    # convert format
    train_df_field = train_df_field[['discharge']]
    train_df_field.index = train_df_field.index.ceil('H')
    val_df_field = val_df_field[['discharge']]
    val_df_field.index = val_df_field.index.ceil('H')

    train_df_field = pp.sample_weights(train_df_field, 'discharge', if_log=True)
    val_df_field = pp.sample_weights(val_df_field, 'discharge', if_log=True)

    # timestamps exist in samples
    timestamps_not_na = df_normed[df_normed['index'].isin(sample_indice)].index

    # only keep tuning targets in sample timestamps
    train_df_field = train_df_field[train_df_field.index.isin(timestamps_not_na)]
    val_df_field = val_df_field[val_df_field.index.isin(timestamps_not_na)]

    # remove duplicates
    train_df_field = train_df_field.groupby(train_df_field.index, as_index=True).mean()
    val_df_field = val_df_field.groupby(val_df_field.index, as_index=True).mean()

    return train_df_field, val_df_field


def get_sequence_indices(train_df_field, val_df_field, df_normed, sample_index):

    train_index_field_df = df_normed[df_normed.index.isin(train_df_field.index)]['index'].values
    val_index_field_df = df_normed[df_normed.index.isin(val_df_field.index)]['index'].values

    train_index_field_sequences = np.where(np.isin(sample_index, train_index_field_df))[0]
    val_index_field_sequences = np.where(np.isin(sample_index, val_index_field_df))[0]

    return train_index_field_sequences, val_index_field_sequences


def adjust_discharge(df_normed, df, device):
    for filename in os.listdir('.'):
        if filename.startswith(f'saved_best_adapted_direct_') and filename.endswith('.pth'):
            rc_pretrain = torch.load(filename)
    df_normed['discharge'] = rc_pretrain(torch.tensor(df_normed['water_level'].values[:, np.newaxis, np.newaxis])
                                         .to(device, dtype=torch.float)).cpu().detach().numpy()[:, 0, 0]
    df['discharge'] = df_normed['discharge'] * (df['discharge'].max() - df['discharge'].min()) + df['discharge'].min()
    return df


def filter_time_df_field(df_field, x_pred_o, time_begin='2016-01-01'):
    df_field['pred_level'] = x_pred_o
    df_field = df_field.reset_index()
    df_field = df_field[df_field['date_time'] > time_begin]  # hard coded
    df_field_index = df_field.index
    x_pred_o = df_field['pred_level'].values[:, np.newaxis]
    df_field = df_field.set_index('date_time')
    return x_pred_o, df_field, df_field_index
