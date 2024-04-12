import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import optuna

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

import warnings
import json
import os
import re

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, data_flood_stage, if_tune
):

    # parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = MinMaxScaler # StandardScaler

    # reload model
    # saved_dir = './outputs/experiments/ARCHIVE_'
    # saved_model = torch.load(
    #     # f'{saved_dir}pi_hodcrnn_1__2024-03-14-17-37-25/best_HODCRNN_optuna_tune_0.00023879233049228787.pth',
    #     # f'{saved_dir}pi_hodcrnn_2__2024-03-14-17-34-33/best_HODCRNN_optuna_tune_0.0004181543772574514.pth',
    #     # f'{saved_dir}pi_hodcrnn_3__2024-03-14-17-32-00/best_HODCRNN_optuna_tune_0.0005952782230451703.pth'
    #     # f'{saved_dir}pi_hodcrnn_4__2024-03-14-17-29-51/best_HODCRNN_optuna_tune_0.0008744496735744178.pth',
    #     # f'{saved_dir}pi_hodcrnn_5__2024-03-14-17-24-40/best_HODCRNN_optuna_tune_0.0010843131458386779.pth',
    #     f'{saved_dir}pi_hodcrnn_6__2024-03-14-17-22-44/best_HODCRNN_optuna_tune_0.001381319249048829.pth',
    # )

    saved_dir = f'./outputs/USGS_{target_gage}'
    saved_folders = os.listdir(f'./outputs/USGS_{target_gage}')
    pre_expr = [i for i in saved_folders if re.match(r"^pi_hodcrnn_\d+", i)]
    assert len(pre_expr) >= 1, 'No expr.'
    pre_expr.sort(reverse=True)
    select_folder = pre_expr[0]
    pretrained_models = [i for i in os.listdir(f'{saved_dir}/{select_folder}')
                         if (i.endswith('.pth')) & (i.startswith('best_'))]
    assert len(pretrained_models) >= 1, 'No pretrain model.'
    pretrained_models.sort()
    pretrained_models_select = pretrained_models[0]
    saved_model = torch.load(
        f'{saved_dir}/{select_folder}/{pretrained_models_select}',
    )

    model = saved_model['model']
    model.eval()
    model.to(device)
    model.name = 'LevelPredHomoDCRNN_tune_base'

    # data
    adj_dis = pd.read_csv(f'{adj_matrix_dir}/adj_matrix.csv', index_col=0)
    num_nodes = len(adj_dis)

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

    # keep usable field measurements (new)
    start_time = df_normed[df_normed['index'] == sequences_w_index[0,0,-1]].index
    df_field = df_field[df_field.index >= start_time.strftime('%Y-%m-%d %H:%M:%S')[0]]
    if len(df_field) < 50:
        warnings.warn(f'Field measurement count is low. {len(df_field)} usable field visits.')

    # index split for major data
    test_percent_updated, test_df_field, num_test_sequences = ft.update_test_percent(df_field, df_normed,
                                                                                     sequences_w_index, test_percent)
    x = sequences_w_index[:, :, :-1][:, :-1, :]
    dataset_index = ft.create_index_4_cv(x, False, None,
                                         val_percent, test_percent_updated, None, None)  # codes for cv is not revised

    # make datasets
    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]
    y = sequences_w_index[:, :, :-1][:, -len(forward):, :]
    y_index = sequences_w_index[:, :, [-1]][:, -len(forward):, :]

    test_x = x[dataset_index[0]['test_index'], :, :][:, :, num_nodes:]
    test_y = y[dataset_index[0]['test_index'], :][:, :, :]
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    if num_test_sequences != test_y.shape[0]:
        raise ValueError('Test sets inconsistency.')

    # pred
    test_pred = mo.pred_4_test_hodcrnn(model, test_x, target_in_forward, device)
    test_pred = test_pred[:, 0, :]

    scaler_pred = scaler()
    scaler_pred.fit(df[[f"{target_gage}_00065"]])
    test_pred = scaler_pred.inverse_transform(pd.DataFrame(test_pred))[:, 0]

    ####################

    # modeled water level
    test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
    test_df = test_df.rename(columns={
        f'{target_gage}_00060': 'modeled',
        f'{target_gage}_00065': 'water_level',
    })
    test_df['pred_water_level'] = test_pred
    test_df_full = test_df.copy()

    # field
    test_df_field.index = test_df_field.index.ceil('H')
    test_df_field = test_df_field.groupby(level=0).mean()
    test_df['field'] = test_df_field['discharge']
    test_df = test_df[~test_df['field'].isna()]
    test_df = test_df.reset_index()

    # base tune
    pred_error_list = ft.remove_test_base_error(test_df, test_df_full, forward)

    test_df['pred_water_level_error'] = pred_error_list
    test_df['pred_water_level_u_tuned'] = test_df['pred_water_level']
    test_df['pred_water_level'] = test_df['pred_water_level'] - test_df['pred_water_level_error']
    test_df['pred_water_level'] = np.round(test_df['pred_water_level'].to_numpy(), 2)
    test_df = test_df.reset_index()
    test_df['pred'] = test_df.apply(
        lambda row: mo.approx_rc(
            row,
            df_raw[[f'{target_gage}_00065', f'{target_gage}_00060']],
            buffer_len=3,
            time_col_name='index', level_col_name='pred_water_level'
        ), axis=1
    )

    return test_df, test_df
