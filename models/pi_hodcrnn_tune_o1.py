import numpy as np
import pandas as pd

import torch
import pickle

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

import warnings
import os
import re


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, data_flood_stage, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = MinMaxScaler # StandardScaler

    # reload model
    # saved_dir = './outputs/experiments/ARCHIVE_'
    # saved_model = torch.load(
    #     # f'{saved_dir}pi_hodcrnn_1__2024-03-14-17-37-25/best_HODCRNN_optuna_tune_0.00023879233049228787.pth',
    #     # f'{saved_dir}pi_hodcrnn_2__2024-03-14-17-34-33/best_HODCRNN_optuna_tune_0.0004181543772574514.pth'
    #     f'{saved_dir}pi_hodcrnn_3__2024-03-14-17-32-00/best_HODCRNN_optuna_tune_0.0005952782230451703.pth',
    #     # f'{saved_dir}pi_hodcrnn_4__2024-03-14-17-29-51/best_HODCRNN_optuna_tune_0.0008744496735744178.pth',
    #     # f'{saved_dir}pi_hodcrnn_5__2024-03-14-17-24-40/best_HODCRNN_optuna_tune_0.0010843131458386779.pth',
    #     # f'{saved_dir}pi_hodcrnn_6__2024-03-14-17-22-44/best_HODCRNN_optuna_tune_0.001381319249048829.pth',
    # )

    saved_dir = f'./outputs/USGS_{target_gage}'
    saved_folders = os.listdir(f'./outputs/USGS_{target_gage}')
    pre_expr = [i for i in saved_folders if re.match(r"^pi_hodcrnn_\d+", i)]
    if len(pre_expr) < 1:
        warnings.warn('No expr.')
        return None, None
    pre_expr.sort(reverse=True)
    select_folder = pre_expr[0]
    pretrained_models = [i for i in os.listdir(f'{saved_dir}/{select_folder}')
                         if (i.endswith('.pth')) & (i.startswith('best_'))]
    if len(pretrained_models) < 1:
        warnings.warn('No pretrain model.')
        return None, None
    pretrained_models.sort()
    pretrained_models_select = pretrained_models[0]
    saved_model = torch.load(
        f'{saved_dir}/{select_folder}/{pretrained_models_select}',
    )

    model = saved_model['model']
    model.eval()
    model.to(device)
    model.name = 'LevelPredHomoDCRNN_tune'

    # data
    df_raw = df.copy()
    df = df.resample('H', closed='right', label='right').mean()
    scaler_stream = scaler()
    df_wl_normed = pd.DataFrame(scaler_stream.fit_transform(df), columns=df.columns, index=df.index)
    wl_cols = [col for col in df.columns if col.endswith('00065')]
    df_wl_normed = df_wl_normed[wl_cols]
    for col in df_wl_normed:
        if col.endswith('00065'):
            df_wl_normed = pp.sample_weights(df_wl_normed, col, if_log=True)

    # precip
    # df_precip_scaled = ft.scale_precip_data(adj_matrix_dir, df_precip)
    # scaler_precip = scaler()
    # df_precip_normed = pd.DataFrame(
    #     scaler_precip.fit_transform(df_precip_scaled), columns=df_precip_scaled.columns, index=df_precip_scaled.index
    # )
    # df_precip_normed = df_precip_normed.rename(columns={0:'ave_precip'})
    assert len(df_precip.columns) == 1, 'Too much cols.'
    scaler_precip = scaler()
    df_precip_normed = pd.DataFrame(
        scaler_precip.fit_transform(df_precip), columns=df_precip.columns, index=df_precip.index
    )
    df_precip_normed = df_precip_normed.rename(columns={df_precip.columns[0]: 'ave_precip'})

    df_normed = pd.concat([
        df_wl_normed,
        df_precip_normed
    ], axis=1)

    # inputs
    target_in_forward = 1
    inputs = (
            sorted([col for col in df_normed if "_weights" in col], reverse=True)
            + sorted([col for col in wl_cols if "_weights" not in col], reverse=True)
            + [df_precip_normed.columns[0]]
    )
    assert inputs[0].split('_')[0] == target_gage, 'Target gage is not at the front!'

    # make sequences and remove samples with nan values
    df_normed['index'] = range(len(df_normed))
    sequences_w_index = ft.create_sequences(df_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # keep usable field measurements
    start_time = df_normed[df_normed['index'] == sequences_w_index[0, 0, -1]].index
    df_field = df_field[df_field.index >= start_time.strftime('%Y-%m-%d %H:%M:%S')[0]]
    if len(df_field) < 50:
        warnings.warn(f'Field measurement count is low. {len(df_field)} usable field visits.')

    # process
    train_x_raw, val_x_raw, test_x_raw, test_y_index, train_df_field, val_df_field, test_df_field = ft.process_tune_data(
        df_field, df_normed,
        sequences_w_index,
        val_percent, test_percent,
        forward,
    )
    x, y, train_index_field, val_index_field = ft.process_tune_data_2(
        df_field, df_normed,
        sequences_w_index,
        val_percent, test_percent,
        forward,
    )

    # pred
    x_pred_o = mo.pred_4_test_hodcrnn(model, x, target_in_forward, device)[:, 0:1, 0].astype('float64')
    scaler_pred = scaler()
    scaler_pred.fit(df[[f"{target_gage}_00065"]])
    x_pred_o = scaler_pred.inverse_transform(pd.DataFrame(x_pred_o))

    train_x_pred_o = x_pred_o[train_index_field, :]
    val_x_pred_o = x_pred_o[val_index_field, :]

    y_denormed = scaler_pred.inverse_transform(pd.DataFrame(y))

    # base tune
    train_x_pred_o = train_x_pred_o - np.array(
        ft.remove_base_error(train_index_field, forward, x_pred_o, y_denormed)
    )[:, np.newaxis]
    val_x_pred_o = val_x_pred_o - np.array(
        ft.remove_base_error(val_index_field, forward, x_pred_o, y_denormed)
    )[:, np.newaxis]

    # # data for tuning: time serie (past values + predicted value) as x / take off data point long time ago
    # train_x_pred_o, train_df_field, train_df_field_index = ft.filter_time_df_field(train_df_field, train_x_pred_o)
    # val_x_pred_o, val_df_field, val_df_field_index = ft.filter_time_df_field(val_df_field, val_x_pred_o)
    # train_x_past = train_x_raw[:, :, 0][train_df_field_index]
    # val_x_past = val_x_raw[:, :, 0][val_df_field_index]

    # format
    train_x_past = train_x_raw[:, :, 0]
    val_x_past = val_x_raw[:, :, 0]
    train_x_past = scaler_pred.inverse_transform(pd.DataFrame(train_x_past))
    val_x_past = scaler_pred.inverse_transform(pd.DataFrame(val_x_past))

    train_x_series = np.concatenate((train_x_past, train_x_pred_o), axis=1)
    val_x_series = np.concatenate((val_x_past, val_x_pred_o), axis=1)
    train_x_series_diff = np.diff(train_x_series, axis=1)
    val_x_series_diff = np.diff(val_x_series, axis=1)

    # convert to dis
    train_x_pred_o = np.round(train_x_pred_o, 2)
    val_x_pred_o = np.round(val_x_pred_o, 2)

    train_x_pred_o_rc = mo.convert_array_w_rc(train_x_pred_o, train_df_field.copy(), df_raw, target_gage)[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(val_x_pred_o, val_df_field.copy(), df_raw, target_gage)[:, np.newaxis]
    # train_x_pred_o_rc = train_x_pred_o_rc_w_na[~np.isnan(train_x_pred_o_rc_w_na)[:, 0], :].copy()
    # val_x_pred_o_rc = val_x_pred_o_rc_w_na[~np.isnan(val_x_pred_o_rc_w_na)[:, 0], :].copy()

    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    # residual error
    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

    train_y_res_per = train_y_res / train_x_pred_o_rc
    val_y_res_per = val_y_res / val_x_pred_o_rc

    # test set
    test_x_pred_o = mo.pred_4_test_hodcrnn(model, test_x_raw, target_in_forward, device)
    test_x_pred_o = test_x_pred_o[:, 0, :].astype(np.float64)
    test_x_pred_o = scaler_pred.inverse_transform(pd.DataFrame(test_x_pred_o))

    test_df_full = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
    test_df_full = test_df_full.rename(columns={
        f'{target_gage}_00060': 'modeled',
        f'{target_gage}_00065': 'water_level',
    })
    test_df_full['pred_water_level'] = test_x_pred_o

    test_df_field.index = test_df_field.index.ceil('H')
    test_df_field = test_df_field.groupby(level=0).mean()
    dt_common = test_df_full.index.intersection(test_df_field.index)
    test_df_field = test_df_field.loc[dt_common]

    test_df_full['field'] = test_df_field['discharge']
    test_df_full['index'] = range(len(test_df_full))
    test_df = test_df_full.loc[~test_df_full['field'].isna(), :].copy()

    test_df_index = test_df['index']
    test_df = test_df.drop('index', axis=1)
    test_df_full = test_df_full.drop('index', axis=1)

    # base tune for test set
    test_df = test_df.reset_index()
    pred_error_list = ft.remove_test_base_error(test_df, test_df_full, forward)
    test_df = test_df.set_index('index')
    test_df['pred_water_level_error'] = pred_error_list
    test_df['pred_water_level_u_tuned'] = test_df['pred_water_level']
    test_df['pred_water_level'] = test_df['pred_water_level'] - test_df['pred_water_level_error']

    # convert
    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_past = test_x_raw[test_df_index.values][:, :, 0]
    test_x_past = (test_x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                   + df[f"{target_gage}_00065"].min())
    test_x_series = np.concatenate((test_x_past, test_x_pred_o), axis=1)
    test_x_series_diff = np.diff(test_x_series, axis=1)

    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o[:, 0], 2),
        test_df.copy(),
        df_raw, target_gage
    )[:, np.newaxis]
    # test_x_pred_o_rc = test_x_pred_o_rc_w_na[~np.isnan(test_x_pred_o_rc_w_na)].copy()

    # residual error
    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field
    test_y_res_per = test_y_res / test_x_pred_o_rc

    # filter - high flow
    # flow_array = np.concatenate((train_x_series[:, -1], val_x_series[:, -1]), axis=0)
    # high_flow = (flow_array.mean() + flow_array.std() * 0.7)  # 6.11
    high_flow = data_flood_stage.iloc[0]['action']
    train_high_flow_index = (train_x_series[:, -2:] >= high_flow).any(axis=1)
    val_high_flow_index = (val_x_series[:, -2:] >= high_flow).any(axis=1)
    test_high_flow_index = (test_x_series[:, -2:] >= high_flow).any(axis=1)

    # filter - high change
    error_rate_array = np.concatenate((
            train_x_series_diff[:, -1] / train_x_series[:, -2],
            val_x_series_diff[:, -1] / val_x_series[:, -2]
    ), axis=0)
    # high_change = (error_rate_array.mean() + error_rate_array.std() * 0.45)  # 0.45
    emergency_ratio = (
            (train_high_flow_index.sum() + val_high_flow_index.sum())
            / (train_high_flow_index.shape[0] + val_high_flow_index.shape[0])
    )
    high_change = np.percentile(error_rate_array, (1-emergency_ratio) * 100)
    train_high_change_index = np.abs(train_x_series_diff[:, -1] / train_x_series[:, -2]) >= high_change
    val_high_change_index = np.abs(val_x_series_diff[:, -1] / val_x_series[:, -2]) >= high_change
    test_high_change_index = np.abs(test_x_series_diff[:, -1] / test_x_series[:, -2]) >= high_change

    # filter
    train_filter = train_high_flow_index & train_high_change_index
    val_filter = val_high_flow_index & val_high_change_index
    test_filter = test_high_flow_index & test_high_change_index

    train_x_series_diff_tune = train_x_series_diff[:, -1][train_filter]
    train_x_series_tune = train_x_series[:, -2][train_filter]
    val_x_series_diff_tune = val_x_series_diff[:, -1][val_filter]
    val_x_series_tune = val_x_series[:, -2][val_filter]
    test_x_series_diff_tune = test_x_series_diff[:, -1][test_filter]
    test_x_series_tune = test_x_series[:, -2][test_filter]

    filter_train_dp = np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0).shape[0]
    if filter_train_dp < 3:
        warnings.warn('Too few training data points for tuning after filtering. Tuning aborted.')
        pp.save_delete_gage_o1_dp(target_gage, forward, 'gauge_delete_o1_dp_few_during_o1')
        return None, None
    if test_x_series_diff_tune.shape[0] < 1:
        warnings.warn('Too few test data points for tuning after filtering. Tuning aborted.')
        pp.save_delete_gage_o1_dp(target_gage, forward, 'gauge_delete_o1_dp_few_during_o1')
        return None, None

    # # vis
    # plt.scatter(
    #     np.concatenate((
    #         train_x_series_diff_tune / train_x_series_tune,
    #         val_x_series_diff_tune / val_x_series_tune,
    #     ), axis=0),
    #     np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0),
    #     color = 'red'
    # )
    # plt.scatter(
    #     test_x_series_diff_tune / test_x_series_tune,
    #     test_y_res_per[test_filter],
    #     color = 'blue'
    # )
    # plt.show()

    # record data
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    train_df_field['last_past_value'] = train_x_past[:, -1]
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    val_df_field['last_past_value'] = val_x_past[:, -1]
    train_val_df_field = pd.concat([train_df_field, val_df_field]).sort_index()
    train_val_df_field.to_csv(f'{expr_dir}/train_val_df.csv')

    # calculate base error ratio
    train_df_field, val_df_field = ft.merge_true_wl_dis(train_df_field, val_df_field, df, target_gage)
    base_ratio = mo.calculate_base_error_ratio(train_df_field, val_df_field, data_flood_stage['action'].iloc[0])

    # residual error learning
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    degree = 1
    model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_train_x = np.concatenate((
            (train_x_series_diff_tune / train_x_series_tune)[:, np.newaxis],
            (val_x_series_diff_tune / val_x_series_tune)[:, np.newaxis],
    ), axis=0)
    model_train_y = np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0)
    index_nan = np.isnan(np.concatenate((model_train_x, model_train_y), axis=1)).any(axis=1)
    model_res.fit(
        model_train_x[~index_nan, :],
        model_train_y[~index_nan, :]
    )
    residual_pred = model_res.predict((test_x_series_diff_tune / test_x_series_tune)[:, np.newaxis])
    residual_pred = residual_pred * test_x_pred_o_rc[test_filter]

    # save and vis
    with open(f'{expr_dir}/tuner_o1_{degree}degree_poly.pkl', 'wb') as f:
        pickle.dump(model_res, f)
    with open(f'{expr_dir}/tuner_o1_apply_index_train.pkl', 'wb') as f:
        pickle.dump(train_filter, f)
    with open(f'{expr_dir}/tuner_o1_apply_index_val.pkl', 'wb') as f:
        pickle.dump(val_filter, f)
    with open(f'{expr_dir}/tuner_o1_apply_index_test.pkl', 'wb') as f:
        pickle.dump(test_filter, f)

    plt.scatter(
        np.concatenate((
            (train_x_series_diff_tune / train_x_series_tune),
            (val_x_series_diff_tune / val_x_series_tune),
        ), axis=0),
        np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0),
        color='red', label='train')
    plt.scatter(
        test_x_series_diff_tune / test_x_series_tune,
        test_y_res_per[test_filter],
        color='blue', label='test')
    x_plot = np.linspace(
        (train_x_series_diff_tune / train_x_series_tune).min(),
        (train_x_series_diff_tune / train_x_series_tune).max(),
        100
    ).reshape(-1, 1)
    y_plot = model_res.predict(x_plot)
    plt.plot(x_plot, y_plot, color='blue')
    plt.show()

    pred_y_tune = test_x_pred_o_rc.copy()
    pred_y_tune[test_filter] = pred_y_tune[test_filter] - residual_pred * (1 - base_ratio)

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc
    test_df.loc[:, 'pred'] = pred_y_tune
    test_df['last_past_value'] = test_x_past[:, -1]

    return test_df, test_df_full
