import pandas as pd
import numpy as np

from scipy.linalg import lstsq
from scipy.stats import pearsonr


def data_import(data_dir, time_interval='15min', data_head_flag='agency_cd', leap_from_flag=1):
    # USE 1: import raw data file downloaded from open sources
    # USE 2: fill the nan values in all df cols using the value of same time in the previous day
    # INPUT: dir of data file
    # OUT: pandas df with missing values as none

    # get the row number of description texts
    with open(data_dir, 'r') as file:
        for num, line in enumerate(file, 1):
            if line.startswith(data_head_flag):
                head_flag = num

    # read file
    data = pd.read_csv(data_dir, skiprows=np.arange(head_flag + leap_from_flag),
                       delimiter='	',
                       names=['USGS', 'code', 'time', 'time_zone', 'value', 'provisional'])
    data = data.drop(['USGS', 'code'], axis=1)

    # remove provisional
    data = data[data['provisional'] == 'A']

    # time format
    data['time'] = pd.to_datetime(data['time'])
    data_est = data[data['time_zone'] == 'EST'].copy()
    data_edt = data[data['time_zone'] == 'EDT'].copy()
    data_est.loc[:, ['time']] = data_est['time'] + pd.to_timedelta(1, unit='h')
    data_est.loc[:, ['time_zone']] = ['EDT'] * len(data_est)
    data = pd.concat([data_est, data_edt], axis = 0).sort_values(by='time')

    # check missing data
    data = data.set_index('time')
    data = data.reindex(pd.date_range(start = data.index[0], end = data.index[-1], freq = time_interval))

    print('Percentage of missing value is', len(data[data.isnull().value]) / len(data) * 100, '%')
    print('Maximum consecutive missing value time period is ',
          (data[data.isna().any(axis = 1)].index.to_series().diff() != pd.Timedelta(
              time_interval)).cumsum().value_counts().max()
          )

    # fill na
    data = data.fillna(data.shift(np.unique(data.index.time).shape[0]))

    return data


def data_imported_combine(*dfs):
    # USE: combine the data imported by data_import
    #      make the datetime index consistent
    # INPUT: a list of df
    # OUTPUT: single df, where each column is one data imported
    #         there will be large chuck of missing values because of datetime index alignment

    # organize data
    data_list = []
    name_list = []
    for d in dfs:
        address = d[0]
        name = d[1]
        data_list.append(data_import(address))
        name_list.append(name)

    end_time_list = [d.index[-1:].to_frame() for d in data_list]
    earliest_end_time = pd.concat(end_time_list, axis=0).min().values[0]
    data_list = [d.loc[: earliest_end_time, :][['value']] for d in data_list]
    data = pd.concat(data_list, axis=1)
    data = data.set_axis(name_list, axis=1)

    return data


def data_add_target(df, sigma_count, forward, target_name ='water_level'):
    # USE: use water level col to determine if there is water level surge
    # INPUT: df w/ water level col
    #        sigma, int, the number of sigma, threshold of water level surge
    #        forward, list of int, time steps for pred
    # OUTPUT: df w/ a target col

    col_diff = df[[target_name]].diff(1)
    threshold = (col_diff.mean() + col_diff.std() * sigma_count).values[0]
    col_diff.loc[col_diff[target_name] >= threshold] = 1
    col_diff.loc[col_diff[target_name] < threshold] = 0

    df['surge'] = col_diff[target_name]
    df = df.iloc[1:]

    return df


def _shift(x, by):
    # if by > 0, nan will be at beginning, first non-nan value was the first value
    # vice versa
    x_shift = np.empty_like(x)
    if by > 0:
        x_shift[:by] = np.nan
        x_shift[by:] = x[:-by]
    else:
        x_shift[by:] = np.nan
        x_shift[:by] = x[-by:]
    return x_shift


def partial_xcorr(x, y, max_lag = 10, standardize = True, reverse = False):
    # Computes partial cross correlation between x and y using linear regression.
    if reverse:
        x = x[::-1]
        y = y[::-1]
    if standardize:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    # Initialize feature matrix
    nlags = max_lag + 1
    X = np.zeros((len(x), nlags))
    X[:, 0] = x
    # Initialize correlations, first is y ~ x
    xcorr = np.zeros(nlags, dtype = float)
    xcorr[0] = pearsonr(x, y)[0]
    # Process lags
    for lag in range(1, nlags):
        # Add lag to matrix
        X[:, lag] = _shift(x, lag)
        # Trim NaNs from y (yt), current lag (l) and previous lags (Z)
        yt = y[lag:]
        l = X[lag:, lag: lag + 1]  # this time lag
        Z = X[lag:, 0: lag]  # all previous time lags
        # Coefficients and residuals for y ~ Z
        beta_l = lstsq(Z, yt)[0]
        resid_l = yt - Z.dot(beta_l)
        # Coefficient and residuals for l ~ Z
        beta_z = lstsq(Z, l)[0]
        resid_z = l - Z.dot(beta_z)
        # Compute correlation between residuals
        xcorr[lag] = pearsonr(resid_l, resid_z.ravel())[0]

    return xcorr
