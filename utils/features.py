import pandas as pd
import numpy as np
from pandas.api.types import is_list_like
from typing import List, Tuple
import warnings


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
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
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


def create_sequences(df, lag, forward, feature_list, target):
    # USE: create sequences with lags for feature and target cols
    # INPUT: df: pandas df
    #        lag: list, starting from 1
    #        feature_list: list of feature names in df
    #        target_list: list of target names in df

    # convert lag and forward to the lag list of furthest target
    lag_update = update_lag_w_forward(lag, forward)

    sequences = np.empty((len(df) - ((lag_update[-1]) + 1) + 1, len(lag_update) + 1, 1))
    for v in ([target] + feature_list):
        df_lag = add_lags(df, lags = lag_update, column = v)[0]
        df_lag = df_lag.dropna()

        drop_list = feature_list + [target]
        drop_list.remove(v)
        df_lag = df_lag.drop(columns = drop_list, axis = 1)

        df_lag = df_lag[df_lag.columns[::-1]]

        sequences = np.concatenate((sequences, df_lag.values[:, :, np.newaxis]), axis = -1)

    sequences = sequences[:, :, 1:]
    return sequences


def _add_time(x, num_time_feature, start_time):
    len_x = x.shape[1]
    for num in range(num_time_feature):
        num_array = np.arange(len_x) + start_time
        num_array_expand = np.expand_dims(np.tile(num_array, [x.shape[0], 1]), axis = 2)
        x = np.concatenate((x, num_array_expand), axis = 2)
        start_time += 1
    return x


def _addLoc(x, numLocFeature, startLoc):
    len_X = x.shape[1]
    for num in range(numLocFeature):
        num_array = np.zeros(len_X) + startLoc
        num_array_expand = np.expand_dims(np.tile(num_array, [x.shape[0], 1]), axis = 2)
        x = np.concatenate((x, num_array_expand), axis = 2)
        startLoc += 1
    return x


def split_sequences(
        sequences,
        val_percent, test_percent, forward,
        random_seed,
        num_time_feature=0, num_loc_feature=0,
        shuffle=False):
    # USE: split the sequences into datasets
    # INPUT: sequences: array, three dims (sample, lag, feature)
    #        val_proportion, test_proportion: float, 0-1
    #        forward: list, time steps forward to predict
    #        num_time_feature, num_loc_feature:
    #        shuffle: bool, if shuffle the sequences
    #        random_seed: int
    # OUTPUT: datasets: array

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(sequences)

    val_count = int(np.floor(sequences.shape[0] * val_percent))
    test_count = int(np.floor(sequences.shape[0] * test_percent))
    train_count = sequences.shape[0] - val_count - test_count

    forward_count = len(forward)

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


# def sequenceTarget(yForward, X):
#     X_out = X[:, :-yForward, :]
#     y_temp = X[:, 1:, 0: 1] # get first one features
#     sequenceList = []
#     for i in range(y_temp.shape[0]):
#         y_tempSliced = y_temp[i, : , :]
#         sequence = sequencesGeneration_update(y_tempSliced, y_tempSliced.shape[0] - 1, yForward)
#         sequenceList.append(sequence)
#     y_out = np.stack(sequenceList)
#     print('X shape', X_out.shape, 'Dims: (sample, sequenceLen, feature)')
#     print('y shape', y_out.shape, 'Dims: (sample, sequenceLen, yForward, feature)')
#     return X_out, y_out
