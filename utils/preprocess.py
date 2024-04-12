import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os
import json
import warnings


def data_import_legacy(data_dir, time_interval='15min', data_head_flag='agency_cd', leap_from_flag=1):
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
    data = pd.concat([data_est, data_edt], axis=0).sort_values(by='time')

    # check missing data
    data = data.set_index('time')
    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq=time_interval))

    print('Percentage of missing value is', len(data[data.isnull().value]) / len(data) * 100, '%')
    print('Maximum consecutive missing value time period is ',
          (data[data.isna().any(axis=1)].index.to_series().diff() != pd.Timedelta(
              time_interval)).cumsum().value_counts().max()
          )

    # fill na
    data = data.fillna(data.shift(np.unique(data.index.time).shape[0]))

    return data


def data_imported_combine_legacy(*dfs):
    # USE: combine the data imported by data_import
    #      make the datetime index consistent
    # INPUT: a list of df
    # OUTPUT: single df, where each column is one data imported
    #         there will be large chuck of missing values because of datetime index alignment

    # organize data
    data_list = []
    name_list = []
    for d in dfs:
        data_list.append(import_data(d[0]))
        name_list.append(d[1])

    end_time_list = [d.index[-1:].to_frame() for d in data_list]
    latest_end_time = pd.concat(end_time_list, axis=0).max().values[0]
    data_list = [d.loc[: latest_end_time, :][['value']] for d in data_list]
    data = pd.concat(data_list, axis=1)
    data = data.set_axis(name_list, axis=1)

    return data


def data_field_import_legacy(dir, after_time = None, data_head_flag='agency_cd', leap_from_flag=1):
    # get the row number of description texts
    with open(dir, 'r') as file:
        for num, line in enumerate(file, 1):
            if line.startswith(data_head_flag):
                head_flag = num

    # read file
    data = pd.read_csv(dir, skiprows=np.arange(head_flag + leap_from_flag), delimiter='	',
                       names=['USGS', 'code', 'meas_index', 'date_time', 'time_zone', 'if_used',
                              'who', 'agency', 'water_level', 'discharge', 'meas_rated', 'level_change',
                              'meas_dura', 'control', 'flow_adjust_code'])
    data = data.drop(['USGS', 'code', 'meas_index', 'who', 'agency',
                      'control', 'flow_adjust_code', 'meas_rated', 'meas_dura'], axis=1)

    # time format
    data['date_time'] = pd.to_datetime(data['date_time'])
    data_est = data[data['time_zone'] == 'EST'].copy()
    data_edt = data[data['time_zone'] == 'EDT'].copy()
    data_est.loc[:, ['date_time']] = data_est['date_time'] + pd.to_timedelta(1, unit='h')
    data_est.loc[:, ['time_zone']] = ['EDT'] * len(data_est)
    data = pd.concat([data_est, data_edt], axis=0).sort_values(by='date_time')
    data = data.drop(['time_zone'], axis=1)

    # filter
    data = data[data['if_used'] == 'Yes']
    data = data.drop(['if_used'], axis=1)
    if after_time:
        data = data[data['date_time'] >= after_time]

    # level adjust
    data['level_change'] = data['level_change'].fillna(0)
    data['water_level_adjusted'] = data['water_level'] + data['level_change'] / 2
    data = data.drop(['water_level', 'level_change'], axis=1)

    # organize
    data = data.set_index('date_time')
    data = data[['water_level_adjusted', 'discharge']]

    return data


def import_data(data_dir, time_interval='15min', tz='America/Chicago'):

    data = pd.read_csv(data_dir)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

    # reset datetime index
    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq=time_interval))
    data.index = data.index.tz_convert(tz)

    # remove provisional
    cd_cols = [col for col in data.columns if col.endswith('_cd')]
    for cd_col in cd_cols:
        data.loc[data[cd_col] == 'P'] = np.nan
    data = data[[col for col in data.columns if 'cd' not in col]]

    # missing value report
    print('Percentage of missing value is', len(data[data.isnull().all(axis=1)]) / len(data) * 100, '%')
    print('Maximum consecutive missing value time period is ',
          (data[data.isna().any(axis=1)].index.to_series().diff() != pd.Timedelta(
              time_interval)).cumsum().value_counts().max()
          )

    # rename cols
    gage_name = data_dir.split('/')[-1].split('.')[0].split('_')[0]
    col_names = [f'{gage_name}_{col}' for col in data.columns]
    data.columns = col_names
    return data


def import_data_simplified(data_dir, time_interval='15min'):

    data = pd.read_csv(data_dir)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

    # select col
    select_col = [col for col in data.columns if col in ['00060', '00065', '00060_cd', '00065_cd']]
    data = data[select_col]

    # remove provisional
    cd_cols = [col for col in data.columns if col.endswith('_cd')]
    for cd_col in cd_cols:
        data = data[data[cd_col] != 'P']
    data = data[[col for col in data.columns if 'cd' not in col]]

    # rename
    data = data.rename(columns = {'00065': 'water_level', '00060': 'discharge'})

    return data


def import_data_combine(dfs, tz, keep_col=['00065', '00060', '00045', '00020']):
    # organize data
    data_list = []
    for d in dfs:
        data_list.append(import_data(d, tz=tz))
    data = pd.concat(data_list, axis=1)

    # select row
    data = data[[col for col in data.columns for k_col in keep_col if k_col in col]]

    return data


def import_data_field(
        dir, after_time = None, data_head_flag='agency_cd', leap_from_flag=1, to_tz=None
):
    # get the row number of description texts
    with open(dir, 'r') as file:
        for num, line in enumerate(file, 1):
            if line.startswith('No sites/data'):
                return None
            if line.startswith(data_head_flag):
                head_flag = num

    # read file
    data = pd.read_csv(dir, skiprows=np.arange(head_flag + leap_from_flag), delimiter='	',
                       names=['USGS', 'code', 'meas_index', 'date_time', 'time_zone', 'if_used',
                              'who', 'agency', 'water_level', 'discharge', 'meas_rated', 'level_change',
                              'meas_dura', 'control', 'flow_adjust_code'])
    data = data.drop(['USGS', 'code', 'meas_index', 'who', 'agency',
                      'control', 'flow_adjust_code', 'meas_rated', 'meas_dura'], axis=1)

    # time format
    data = data[data['date_time'].str.len() > 10]
    data['date_time'] = pd.to_datetime(data['date_time'])
    data = data.set_index('date_time')

    # time zone
    tz_mapping = {
        'EDT': 'America/New_York',
        'EST': 'America/New_York',
        'CDT': 'America/Chicago',
        'CST': 'America/Chicago',
        'MDT': 'America/Denver',
        'MST': 'America/Denver',
        'PDT': 'America/Los_Angeles',
        'PST': 'America/Los_Angeles',
        'AKDT': 'America/Anchorage',
        'AKST': 'America/Anchorage',
        'HDT': 'America/Adak',
        'HST': 'America/Adak',
        'AST': 'America/Puerto_Rico',
        'ADT': 'America/Halifax',
        'SST': 'Pacific/Pago_Pago',
        'ChST': 'Pacific/Guam',
    }
    tz_abbr = data['time_zone'].unique().tolist()
    tz_name = [tz_mapping[tz] for tz in tz_abbr]
    if len(set(tz_name)) <= 1:
        data.index = data.index.tz_localize(tz_name[0])
        data = data.drop(['time_zone'], axis=1)
    else:
        raise ValueError(f"Conflicting time zone for gage {dir.split('/')[-1].split('.')[0]}")

    # filter
    data = data[data['if_used'] == 'Yes']
    data = data.drop(['if_used'], axis=1)
    if after_time:
        data = data[data['date_time'] >= after_time]

    # level adjust
    data['level_change'] = data['level_change'].fillna(0)
    data['water_level_adjusted'] = data['water_level'] + data['level_change'] / 2
    data = data.drop(['water_level', 'level_change'], axis=1)

    # organize
    data = data[['water_level_adjusted', 'discharge']]

    # change tz
    if to_tz is not None:
        data.index = data.index.tz_convert(to_tz)

    return data


def import_data_rc(dir, data_head_flag='INDEP	SHIFT	DEP	STOR', leap_from_flag=1):
    # get the row number of description texts
    head_flag = None
    with open(dir, 'r') as file:
        for num, line in enumerate(file, 1):
            if line.startswith(data_head_flag):
                head_flag = num

    if head_flag is None:
        return None

    # read file
    data = pd.read_csv(
        dir, skiprows=np.arange(head_flag + leap_from_flag), delimiter='	',
        names=['water_level', 'shift', 'discharge', 'asterisk'], index_col=None)
    data = data.drop(['asterisk', 'shift'], axis=1)

    return data.set_index('water_level')['discharge'].to_dict()


def import_data_shift_rc_0157360(curve, shift, data_head_flag='# Gage height (ft)	Discharge (ft^3/s)'):
    rc_file_name = f"./data/USGS_gage_01573560_shift_curves/01573560 shift {curve}_{shift}.txt"

    head_flag = None
    with open(rc_file_name, 'r') as file:
        for num, line in enumerate(file, 1):
            if line.startswith(data_head_flag):
                head_flag = num
    if head_flag is None:
        return None

    rc = pd.read_csv(
        rc_file_name, skiprows=np.arange(head_flag), delimiter='	',
        names=['gauge height', 'discharge'], index_col=None)

    return rc

def import_data_precipitation_legacy(dir, lat_list, lon_list, tz):
    csv_names = [file.split('.csv')[0] for file in os.listdir(dir) if file.endswith('.csv')]
    df_list = []
    for lat in lat_list:
        for lon in lon_list:
            if f'clat{lat}_clon{lon}' in csv_names:
                df = pd.read_csv(f'{dir}/clat{lat}_clon{lon}.csv')
                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].dt.tz_convert(tz)
                df = df.set_index('date')
                df = df.rename(columns={'value': f'clat{lat}_clon{lon}'})
                df_list.append(df)
    df_precip = pd.concat(df_list, axis=1)

    # take off abnormal values
    df_precip[df_precip < 0] = np.nan
    df_precip[df_precip > 100] = np.nan
    return df_precip


def import_data_precipitation(dir, gauge, tz):
    csv_names = [file.split('.csv')[0] for file in os.listdir(dir) if file.endswith('.csv')]
    csv_select = [c for c in csv_names if c.startswith(f'USGS_{gauge}')]
    assert len(csv_select) == 1, 'Duplicated or missing file.'

    df = pd.read_csv(f'{dir}/{csv_select[0]}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_convert(tz)
    df_precip = df.set_index('date')

    # take off abnormal values
    df_precip[df_precip < 0] = np.nan
    df_precip[df_precip > 100] = np.nan
    return df_precip


def data_add_target(df, sigma_count, forward, target_name='water_level'):
    # USE: use water level col to determine if there is water level surge
    # INPUT: df w/ water level col
    #        sigma, int, the number of sigma, threshold of water level surge
    #        forward, list of int, time steps for pred
    # OUTPUT: df w/ a target col

    col_diff = df[[target_name]].diff(1)
    col_diff_01 = col_diff.copy()
    threshold = (col_diff_01.mean() + col_diff_01.std() * sigma_count).values[0]
    col_diff_01.loc[col_diff_01[target_name] < threshold] = 0
    col_diff_01.loc[col_diff_01[target_name] >= threshold] = 1

    df['water_level_diff'] = col_diff
    df['surge'] = col_diff_01[target_name]
    df = df.iloc[1:]

    return df


def data_add_water_level_diff(df, discharge_diff=False):
    # USE: add the diff of water level as a col
    # INPUT: df
    # OUTPUT: df

    col_diff = df[['water_level']].diff(1)
    df['water_level_diff'] = col_diff

    if discharge_diff:
        col_diff = df[['discharge']].diff(1)
        df['discharge_diff'] = col_diff

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


def partial_xcorr(x, y, max_lag=10, standardize=True, reverse=False):
    # Computes partial cross correlation between x and y using linear regression.

    from scipy.linalg import lstsq
    from scipy.stats import pearsonr

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
    xcorr = np.zeros(nlags, dtype=float)
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


def sample_weights(df, col, num_bin=10, if_log=False):
    df_clean = df.dropna().copy()
    bin_counts, bin_edges = np.histogram(df_clean[col], bins=num_bin)
    bin_indices = np.digitize(df_clean[col], bin_edges[:-1])
    bin_counts_altered = bin_counts.copy()
    bin_counts_altered[bin_counts_altered < 2] = 2
    if if_log:
        bin_counts_altered = np.log(bin_counts_altered)
    weights = bin_counts_altered.mean() / bin_counts_altered[bin_indices - 1]
    df_clean[col + '_weights'] = weights
    df = df.merge(df_clean[[col + '_weights']], how='left', left_index=True, right_index=True)
    return df


def pull_USGS_gage_iv(working_dir, USGS_gauges, skip_gauge=[], start='2018-12-24', end='2023-12-24',
                      feature_list=['00065', '00060', '00045', '00020']):

    import dataretrieval.nwis as nwis

    exist_files = [os.path.splitext(f)[0] for f in os.listdir(working_dir) if os.path.splitext(f)[1] == '.csv']
    download_files = [f for f in USGS_gauges['SITENO'].to_list() if f not in exist_files]
    print(f'{len(exist_files)} iv files exist.')
    print(f'{len(download_files)} iv files to download, including skipped ones.')
    for gg in download_files:
        if gg not in skip_gauge:  # error
            try:
                df_iv = nwis.get_record(sites=gg, service='iv', start=start, end=end)
                # 00065: gauging height; 00060: discharge; 00045: precipitation; 00020: air temperature
                if feature_list != []:
                    select_col = [col for col in feature_list if col in df_iv.columns]
                else:
                    select_col = df_iv.columns
                df_iv = df_iv[select_col]
                df_iv.to_csv(f'{working_dir}/{gg}.csv')
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    return


def pull_USGS_gage_flood_stage(working_dir, USGS_gauges):
    import xml.etree.ElementTree as ET
    import requests

    df_list = []
    if os.path.exists(f'{working_dir}/flood_stages.csv'):
        saved_df = pd.read_csv(f'{working_dir}/flood_stages.csv',
                               index_col=0, dtype={'site_no': str})
        df_list.append(saved_df)
        exist_sites = saved_df['site_no'].to_list()
    else:
        exist_sites = []
    download_sites = [f for f in USGS_gauges['SITENO'].to_list() if f not in exist_sites]
    print(f'Flooding stages of {len(exist_sites)} sites exist.')
    print(f'{len(download_sites)} sites to pull.')
    for gg in download_sites:
        try:
            url = f'https://waterwatch.usgs.gov/webservices/floodstage?site={gg}&format=xml'
            response = requests.get(url)

            xml_data = response.content
            root = ET.fromstring(xml_data)
            site_no = root.find('.//site_no')
            action_stage = site_no.find('action_stage').text if site_no is not None else None
            flood_stage = site_no.find('flood_stage').text if site_no is not None else None
            moderate_stage = site_no.find('moderate_flood_stage').text if site_no is not None else None
            major_stage = site_no.find('major_flood_stage').text if site_no is not None else None
            df = pd.DataFrame({
                'site_no': [gg],
                'action': [action_stage],
                'flood': [flood_stage],
                'moderate': [moderate_stage],
                'major': [major_stage],
            })
            df_list.append(df)

        except Exception as e:
            print(f"An error occurred for {gg}: {e}")
            continue
    df = pd.concat(df_list, axis=0)
    df.to_csv(f'{working_dir}/flood_stages.csv')
    return


def count_USGS_gage_flood(working_dir_USGS_iv, USGS_gauges, working_dir_USGS_major_flood_riv, skip=[]):
    saved_gage = [file.split('.')[0] for file in os.listdir(working_dir_USGS_iv) if file.endswith('.csv')]
    for gauging_name in USGS_gauges['SITENO']:

        if gauging_name not in saved_gage:
            continue
        if gauging_name in skip:
            continue

        records = pd.read_csv(f'{os.path.join(working_dir_USGS_iv, gauging_name)}.csv')
        records['datetime'] = pd.to_datetime(records['datetime'])
        records = records.set_index('datetime')
        records = records.resample('D').max()

        try:
            gauging_heights = records[['00065']]
        except Exception as e:
            print(f'{gauging_name} does not have gauging height records.')
            continue

        action_count = int(
            (gauging_heights > float(USGS_gauges[USGS_gauges['SITENO'] == gauging_name]['action'].values[0]))
            .any(axis=1).sum())
        flood_count = int(
            (gauging_heights > float(USGS_gauges[USGS_gauges['SITENO'] == gauging_name]['flood'].values[0]))
            .any(axis=1).sum())
        moderate_count = int(
            (gauging_heights > float(USGS_gauges[USGS_gauges['SITENO'] == gauging_name]['moderate'].values[0]))
            .any(axis=1).sum())
        major_count = int(
            (gauging_heights > float(USGS_gauges[USGS_gauges['SITENO'] == gauging_name]['major'].values[0]))
            .any(axis=1).sum())

        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'action_count'] = action_count
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'flood_count'] = flood_count
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'moderate_count'] = moderate_count
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'major_count'] = major_count

    USGS_gauges.to_csv(working_dir_USGS_major_flood_riv + '/gauge_flood_counts.csv', index=False)
    return USGS_gauges


def pull_USGS_gage_field(working_dir_USGS_field, USGS_gauges):
    import requests
    exist_files_field = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_field)]
    download_files_field = [f for f in USGS_gauges['SITENO'].to_list() if f not in exist_files_field]
    print(f'{len(exist_files_field)} field data exist.')
    print(f'{len(download_files_field)} field data to download.')

    for gg in download_files_field:
        url = f'https://waterdata.usgs.gov/nwis/measurements?site_no={gg}&agency_cd=USGS&format=rdb'
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{working_dir_USGS_field}/{gg}.csv', 'wb') as file:
                file.write(response.content)
    return


def pull_USGS_gage_rc(working_dir_USGS_rc, USGS_gauges):
    import requests
    exist_files_rc = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_rc)]
    download_files_rc = [f for f in USGS_gauges['SITENO'].to_list() if f not in exist_files_rc]
    print(f'{len(exist_files_rc)} rc data exist.')
    print(f'{len(download_files_rc)} rc data to download.')

    for gg in download_files_rc:
        url = f'https://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={gg}'
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{working_dir_USGS_rc}/{gg}_rc.txt', 'wb') as file:
                file.write(response.content)
    return


def count_USGS_gage_field(working_dir_USGS_field, working_dir_USGS_iv, working_dir_USGS_flood_stage, USGS_gauges):
    flood_stage_file = pd.read_csv(
        working_dir_USGS_flood_stage + './flood_stages.csv',
        dtype={'site_no': 'str'}
    )
    exist_files_field = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_field)]
    for gauging_name in USGS_gauges['SITENO']:
        if gauging_name not in exist_files_field:
            print(f'Data for {gauging_name} is missing.')
            continue

        get_header_line = False
        with open(f'{working_dir_USGS_field}/{gauging_name}.csv', 'r') as file:
            for i, line in enumerate(file):
                if 'agency_cd' in line:
                    header_line = i
                    get_header_line = True
                    break

        if get_header_line == False:
            print(f'No field measurement for gauge {gauging_name}')
            continue

        field_measures = import_data_field(f'{working_dir_USGS_field}/{gauging_name}.csv')
        iv_measures = import_data_simplified(f'{working_dir_USGS_iv}/{gauging_name}.csv')

        field_measures_count = 0
        if field_measures is not None and "water_level" in iv_measures.columns:
            iv_measures = iv_measures[iv_measures["water_level"] != -999999]
            field_measures.index = field_measures.index.round("15min")
            data_field_modeled = field_measures.merge(
                iv_measures,
                left_on=field_measures.index,
                right_on=iv_measures.index,
                how="inner",
                suffixes=("_field", "_modeled"),
            ).dropna()
            field_measures_count = len(data_field_modeled)

            USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'field_measure_count'] = field_measures_count

            # extra count
            try:
                action_stage = flood_stage_file.loc[flood_stage_file['site_no'] == gauging_name, 'action'].values[0]
                flood_stage = flood_stage_file.loc[flood_stage_file['site_no'] == gauging_name, 'flood'].values[0]
                moderate_stage = flood_stage_file.loc[flood_stage_file['site_no'] == gauging_name, 'moderate'].values[0]
                major_stage = flood_stage_file.loc[flood_stage_file["site_no"] == gauging_name, 'major'].values[0]

                USGS_gauges.loc[
                    USGS_gauges["SITENO"] == gauging_name, "field_measure_count_action"
                ] = len(data_field_modeled[data_field_modeled['water_level'] >= action_stage])
                USGS_gauges.loc[
                    USGS_gauges["SITENO"] == gauging_name, "field_measure_count_flood"
                ] = len(data_field_modeled[data_field_modeled['water_level'] >= flood_stage])
                USGS_gauges.loc[
                    USGS_gauges["SITENO"] == gauging_name, "field_measure_count_moderate"
                ] = len(data_field_modeled[data_field_modeled['water_level'] >= moderate_stage])
                USGS_gauges.loc[
                    USGS_gauges["SITENO"] == gauging_name, "field_measure_count_major"
                ] = len(data_field_modeled[data_field_modeled['water_level'] >= major_stage])
            except:
                pass

    return USGS_gauges


def pull_USGS_up_gage(working_dir_USGS_upstream, USGS_gauges, search_distance=10, scope='UT'):
    import requests

    exist_files = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_upstream)]
    exist_files = [f for f in exist_files if f.endswith(f'_s{search_distance}_{scope}')]
    download_files = [f for f in USGS_gauges['SITENO'].to_list() if
                      f'{f}_s{search_distance}_{scope}' not in exist_files]
    print(f'{len(exist_files)} upstream gage data with given search distance/scope exist.')
    print(f'{len(download_files)} upstream gage data to download.')
    for gg in download_files:
        try:
            url = (f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{gg}/" +
                   f"navigation/{scope}/nwissite?f=json&distance={search_distance}")
            response = requests.get(url)
            geojson_data = response.json()
            with open(f'{working_dir_USGS_upstream}/{gg}_s{search_distance}_{scope}.geojson', 'w') as file:
                json.dump(geojson_data, file, indent=4)
        except:
            print(f'Data pull failed for gage {gg}.')
            if response.status_code != 200:
                print(f'Fail code: {response.status_code}.')
    return


def count_USGS_up_gage(working_dir_USGS_upstream, USGS_gauges, search_distance=10, scope='UT'):
    import geopandas as gpd

    exist_files = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_upstream)]
    exist_files = [f for f in exist_files if f.endswith(f'_s{search_distance}_{scope}')]
    for gauging_name in USGS_gauges['SITENO']:
        if f'{gauging_name}_s{search_distance}_{scope}' not in exist_files:
            print(f'Data for {gauging_name} is missing.')
            continue
        upstream_gages = gpd.read_file(f'{working_dir_USGS_upstream}/{gauging_name}_s{search_distance}_{scope}.geojson')
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name,
        f'{search_distance}_{scope}_upstream_gage_count'] = len(upstream_gages)
    return USGS_gauges


def check_USGS_gage_feature(working_dir_USGS_iv, USGS_gauges):
    saved_gage = [file.split('.')[0] for file in os.listdir(working_dir_USGS_iv) if file.endswith('.csv')]
    for gauging_name in USGS_gauges['SITENO']:
        if gauging_name not in saved_gage:
            continue

        records = pd.read_csv(f'{os.path.join(working_dir_USGS_iv, gauging_name)}.csv', nrows=10)
        records['datetime'] = pd.to_datetime(records['datetime'])
        records = records.set_index('datetime')
        num_cols = len(records.columns)

        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'num_features'] = num_cols
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name, 'features'] = ', '.join(records.columns)

    return USGS_gauges


def check_USGS_up_gage_feature(working_dir_USGS_iv, working_dir_USGS_upstream, USGS_gauges, search_distance=50,
                               check_on='UT', check_for=['00020', '00045']):

    import geopandas as gpd

    exist_files = [os.path.splitext(f)[0] for f in os.listdir(working_dir_USGS_upstream)]
    exist_files = [f for f in exist_files if f.endswith(f'_s{search_distance}_{check_on}')]
    for gauging_name in USGS_gauges['SITENO']:
        if f'{gauging_name}_s{search_distance}_{check_on}' not in exist_files:
            print(f'Data for {gauging_name} is missing.')
            continue
        upstream_gages = gpd.read_file(
            f'{working_dir_USGS_upstream}/{gauging_name}_s{search_distance}_{check_on}.geojson')

        num_preci_temprs = []
        for identifier in upstream_gages['identifier']:
            id = identifier.replace('USGS-', '')

            try:
                up_gage_records = pd.read_csv(f'{os.path.join(working_dir_USGS_iv, id)}.csv', nrows=10)
            except:
                USGS_gauges_temp = pd.DataFrame({'SITENO': [id]})
                print(f'{id} iv file is missing. Pull it.')
                pull_USGS_gage_iv(working_dir_USGS_iv, USGS_gauges_temp, start='2022-12-30', end='2022-12-31')
                up_gage_records = pd.read_csv(f'{os.path.join(working_dir_USGS_iv, id)}.csv', nrows=10)
            try:
                up_gage_records['datetime'] = pd.to_datetime(up_gage_records['datetime'])
                up_gage_records = up_gage_records.set_index('datetime')
                num_preci_tempr = sum(col in check_for for col in up_gage_records.columns)
                num_preci_temprs.append(num_preci_tempr)
            except:
                num_preci_temprs.append(0)

        tag_fear = [str(int(i)) for i in check_for]
        USGS_gauges.loc[USGS_gauges['SITENO'] == gauging_name,
        f"ave_num_feat_{'_'.join(tag_fear)}"] = sum(num_preci_temprs) / len(num_preci_temprs)
    return USGS_gauges


def pull_USGS_stream_geo(working_dir, USGS_gauges_ID_list, on='UT', dist=50):
    import requests
    for gg in USGS_gauges_ID_list:
        try:
            url = (f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{gg}/navigation/{on}/flowlines?distance={dist}")
            response = requests.get(url)
            geojson_data = response.json()
            with open(f'{working_dir}/{gg}_s{dist}_{on}_stream.geojson', 'w') as file:
                json.dump(geojson_data, file, indent=4)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return


def pull_USGS_gage_geo(working_dir, USGS_gauges_ID_list):
    import requests
    exist_files = [
        os.path.splitext(f)[0].split('_geo')[0] for f in os.listdir(working_dir) if f.endswith('.geojson')
    ]
    for gg in USGS_gauges_ID_list:
        if gg in exist_files:
            print(f'Geo info of {gg} exists.')
            continue
        else:
            print(f'Pull geo info of {gg}.')
            try:
                url = (f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{gg}?f=json")
                response = requests.get(url)
                geojson_data = response.json()
                with open(f'{working_dir}/{gg}_geo.geojson', 'w') as file:
                    json.dump(geojson_data, file, indent=4)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    return


def merge_field_modeled(data_field, data):
    if 'discharge' not in data.columns:
        return None

    # clean
    data = data[data['discharge'] != -999999]

    # remove field measurements that are too low
    data_field = data_field[data_field['discharge'] >= 1]

    # field measurement vs modeled
    data_field.index = data_field.index.round("15min")
    data_field_modeled = data_field.merge(data, left_on=data_field.index, right_on=data.index,
                                          how='inner', suffixes=('_field', '_modeled')).dropna()
    data_field_modeled = data_field_modeled.set_index('key_0')
    data_field_modeled['perc_error'] = ((data_field_modeled['discharge_modeled'] -
                                         data_field_modeled['discharge_field']).abs() /
                                        data_field_modeled['discharge_field']).fillna(0) * 100
    return data_field_modeled


def list_depth(L):
    if isinstance(L, list):
        if not L:
            return 1
        return 1 + max(list_depth(item) for item in L)
    else:
        return 0

def get_bounding_grid(watershed):
    import math

    watershed = watershed["features"][0]["geometry"]["coordinates"]
    num_depth = list_depth(watershed)

    for i in range(num_depth - 2):
        watershed = watershed[0]

    watershed_lat = [l[1] for l in watershed]
    watershed_lon = [l[0] for l in watershed]
    watershed_lat_min = min(watershed_lat)
    watershed_lat_max = max(watershed_lat)
    watershed_lon_min = min(watershed_lon)
    watershed_lon_max = max(watershed_lon)
    b_lat_min = math.floor(watershed_lat_min * 10) / 10.0
    b_lat_max = math.floor(watershed_lat_max * 10) / 10.0
    b_lon_min = math.floor(watershed_lon_min * 10) / 10.0
    b_lon_max = math.floor(watershed_lon_max * 10) / 10.0

    lat_list = list(np.arange(b_lat_min, b_lat_max + 0.1, 0.1))
    lat_list = [str(round(num, 1)) for num in lat_list]
    lon_list = list(np.arange(b_lon_min, b_lon_max + 0.1, 0.1))
    lon_list = [str(round(num, 1)) for num in lon_list]

    return lat_list, lon_list


def get_bounds(watershed):
    import math

    watershed = watershed["features"][0]["geometry"]["coordinates"]
    num_depth = list_depth(watershed)

    for i in range(num_depth - 2):
        watershed = watershed[0]

    watershed_lat = [l[1] for l in watershed]
    watershed_lon = [l[0] for l in watershed]
    watershed_lat_min = min(watershed_lat)
    watershed_lat_max = max(watershed_lat)
    watershed_lon_min = min(watershed_lon)
    watershed_lon_max = max(watershed_lon)
    b_lat_min = math.floor(watershed_lat_min * 10) / 10.0
    b_lat_max = math.ceil(watershed_lat_max * 10) / 10.0
    b_lon_min = math.floor(watershed_lon_min * 10) / 10.0
    b_lon_max = math.ceil(watershed_lon_max * 10) / 10.0

    return b_lat_min, b_lat_max, b_lon_min, b_lon_max


def count_o1_dp(
        df, df_field, test_df_field, sequences_w_index, target_gage, forward, data_flood_stage
):
    # filter out gage w/o enough field measurements for o1 during test period
    field_test = test_df_field.copy()
    field_test.index = field_test.index.ceil('H')
    field_test = field_test.groupby(level=0).mean()
    field_train_val = df_field[~df_field.index.isin(test_df_field.index)].copy()
    field_train_val.index = field_train_val.index.ceil('H')
    field_train_val = field_train_val.groupby(level=0).mean()

    df_filter = df.copy()
    df_filter = df_filter.reset_index()
    df_mask = df_filter.index.isin(sequences_w_index[:, -1, -1].astype(int))
    df_filter.iloc[~df_mask, 1:] = np.nan
    df_filter = df_filter.set_index('index')

    wl_all = df_filter[[f'{target_gage}_00065']]
    wl_test = wl_all.loc[wl_all.index.isin(field_test.index)]
    wl_test_earlier = wl_all.loc[wl_all.index.isin(field_test.index - pd.Timedelta(hours=forward[0]))]
    wl_train_val = wl_all.loc[wl_all.index.isin(field_train_val.index)]
    wl_train_val_earlier = wl_all.loc[wl_all.index.isin(field_train_val.index - pd.Timedelta(hours=forward[0]))]
    assert len(wl_train_val) == len(wl_train_val_earlier), 'df len inconsistent.'
    assert len(wl_test) == len(wl_test_earlier), 'df len inconsistent.'

    # train val
    wl_train_val = wl_train_val.reset_index()
    wl_train_val_earlier = wl_train_val_earlier.reset_index()
    nan_index_train_val = wl_train_val[wl_train_val.isna().any(axis=1)].index.union(
        wl_train_val_earlier[wl_train_val_earlier.isna().any(axis=1)].index
    )
    wl_train_val = wl_train_val[~wl_train_val.index.isin(nan_index_train_val)]
    wl_train_val_earlier = wl_train_val_earlier[~wl_train_val_earlier.index.isin(nan_index_train_val)]
    wl_train_val = wl_train_val.set_index('index')
    wl_train_val_earlier = wl_train_val_earlier.set_index('index')

    wl_train_val_cat = np.concatenate(
        (wl_train_val[[f'{target_gage}_00065']].values, wl_train_val_earlier[[f'{target_gage}_00065']].values),
        axis= 1
    )
    index_high_flow_train_val = (wl_train_val_cat >= data_flood_stage['action'].iloc[0]).any(axis=1)
    em_ratio = index_high_flow_train_val.sum() / wl_train_val.shape[0]
    diff_flow_train_val = np.abs(
            wl_train_val[[f'{target_gage}_00065']].values
            - wl_train_val_earlier[[f'{target_gage}_00065']].values
    )
    high_change_train_val = np.percentile(
        diff_flow_train_val,
        (1 - em_ratio) * 100,
    )
    index_high_change_train_val = np.abs(
            wl_train_val[[f'{target_gage}_00065']].values - wl_train_val_earlier[[f'{target_gage}_00065']].values
    ) >= high_change_train_val
    index_select_o1_train_val = index_high_flow_train_val & index_high_change_train_val[:, 0]
    o1_dp_train_val = index_select_o1_train_val.sum()

    # test
    wl_test = wl_test.reset_index()
    wl_test_earlier = wl_test_earlier.reset_index()
    nan_index_test = wl_test[wl_test.isna().any(axis=1)].index.union(
        wl_test_earlier[wl_test_earlier.isna().any(axis=1)].index
    )
    wl_test = wl_test[~wl_test.index.isin(nan_index_test)]
    wl_test_earlier = wl_test_earlier[~wl_test_earlier.index.isin(nan_index_test)]
    wl_test = wl_test.set_index('index')
    wl_test_earlier = wl_test_earlier.set_index('index')

    wl_test_cat = np.concatenate(
        (wl_test[[f'{target_gage}_00065']].values, wl_test_earlier[[f'{target_gage}_00065']].values),
        axis= 1
    )
    index_high_flow_test = (wl_test_cat >= data_flood_stage['action'].iloc[0]).any(axis=1)
    diff_flow_test = np.abs(
            wl_test[[f'{target_gage}_00065']].values
            - wl_test_earlier[[f'{target_gage}_00065']].values
    )
    high_change_test = np.percentile(
        diff_flow_test,
        (1 - em_ratio) * 100,
    )
    index_high_change_test = np.abs(
            wl_test[[f'{target_gage}_00065']].values - wl_test_earlier[[f'{target_gage}_00065']].values
    ) >= high_change_test
    index_select_o1_test = index_high_flow_test & index_high_change_test[:, 0]
    o1_dp_test = index_select_o1_test.sum()
    return o1_dp_train_val, o1_dp_test


def save_delete_gage_o1_dp(target_gage, forward):
    warnings.warn(f'Field measurement for o1 is low.')
    if os.path.exists(f'./outputs/USGS_gaga_filtering/gauge_delete_o1_dp_few_{forward[0]}.json'):
        with open(f'./outputs/USGS_gaga_filtering/gauge_delete_o1_dp_few_{forward[0]}.json', 'r') as f:
            list_few_field_test = json.load(f)
    else:
        list_few_field_test = []
    list_few_field_test = list_few_field_test + [target_gage]
    list_few_field_test = list(set(list_few_field_test))
    with open(f'./outputs/USGS_gaga_filtering/gauge_delete_o1_dp_few_{forward[0]}.json', 'w') as f:
        json.dump(list_few_field_test, f)
