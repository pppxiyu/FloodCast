import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp
from utils.features import process_tune_data
from models.pi_hodcrnn_tune_o2 import apply_o1_tuner, apply_o1_tune_4_test
import scipy.interpolate as interpolate

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

import warnings

def apply_o2_tuner(x_pred_o_rc, x_pred_o, tuner_o2_list, base_ratio=0, if_update=True):
    residual_pred = np.full(x_pred_o_rc.shape, np.nan)
    for tuner_o2 in tuner_o2_list:
        interpolation_function = interpolate.interp1d(tuner_o2[:, 0], tuner_o2[:, 1])
        apply_index = (x_pred_o <= tuner_o2[:, 0].max()) & (x_pred_o >= tuner_o2[:, 0].min())
        residual_pred[apply_index] = interpolation_function(x_pred_o[apply_index])

        # handle the out-of-range points
        residual_pred_t = np.full(x_pred_o[~apply_index].shape, np.nan)
        i = 0
        for v in x_pred_o[~apply_index]:
            differences = np.abs(tuner_o2[:, 0] - v)
            closest_x = tuner_o2[:, 0][np.argpartition(differences, 2)[:2]]
            closest_y = tuner_o2[:, 1][np.argpartition(differences, 2)[:2]]
            v_y = ((v - closest_x[1]) * (closest_y[0] - closest_y[1]) / (closest_x[0] - closest_x[1])) + closest_y[1]
            residual_pred_t[i] = v_y
        residual_pred[~apply_index] = residual_pred_t

    assert ~np.isnan(residual_pred).any(), 'Missing prediction.'
    residual_pred = residual_pred * x_pred_o_rc
    if if_update:
        train_x_pred_o_rc_updated = x_pred_o_rc - residual_pred * (1 - base_ratio)
        return train_x_pred_o_rc_updated
    else:
        return residual_pred


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, data_flood_stage, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = MinMaxScaler # StandardScaler

    # reload model
    saved_model = torch.load(
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_1__2024-03-14-17-37-25/best_HODCRNN_optuna_tune_0.00023879233049228787.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_2__2024-03-14-17-34-33/best_HODCRNN_optuna_tune_0.0004181543772574514.pth'
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_3__2024-03-14-17-32-00/best_HODCRNN_optuna_tune_0.0005952782230451703.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_4__2024-03-14-17-29-51/best_HODCRNN_optuna_tune_0.0008744496735744178.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_5__2024-03-14-17-24-40/best_HODCRNN_optuna_tune_0.0010843131458386779.pth',
        './outputs/experiments/ARCHIVE_pi_hodcrnn_6__2024-03-14-17-22-44/best_HODCRNN_optuna_tune_0.001381319249048829.pth',
    )
    model = saved_model['model']
    model.eval()
    model.to(device)
    model.name = 'LevelPredHomoDCRNN_tune'

    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_1__2024-04-04-13-15-56'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_2__2024-04-04-13-26-18'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_3__2024-04-04-13-27-54'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_4__2024-04-04-13-29-34'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_5__2024-04-04-13-30-53'
    dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_6__2024-04-04-13-32-38'
    with open(f'{dir_o1}/tuner_o1_1degree_poly.pkl', 'rb') as file:
        tuner_o1 = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_train.pkl', 'rb') as file:
        tuner_o1_train_index = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_val.pkl', 'rb') as file:
        tuner_o1_val_index = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_test.pkl', 'rb') as file:
        tuner_o1_test_index = pickle.load(file)

    tuner_o2_list = []
    # dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_1__2024-04-04-17-28-57'
    # dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_2__2024-04-04-17-32-52'
    # dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_3__2024-04-04-17-36-10'
    # dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_4__2024-04-04-17-38-12'
    # dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_5__2024-04-04-17-40-03'
    dir_o2 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o2_6__2024-04-04-17-41-39'
    with open(f'{dir_o2}/tuner_o2_lowess.pkl', 'rb') as file:
        tuner_o2_list.append(pickle.load(file))

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
    df_precip_scaled = ft.scale_precip_data(adj_matrix_dir, df_precip)
    scaler_precip = scaler()
    df_precip_normed = pd.DataFrame(
        scaler_precip.fit_transform(df_precip_scaled), columns=df_precip_scaled.columns, index=df_precip_scaled.index
    )
    df_precip_normed = df_precip_normed.rename(columns={0:'ave_precip'})

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
    start_time = df_normed[df_normed['index'] == sequences_w_index[0,0,-1]].index
    df_field = df_field[df_field.index >= start_time.strftime('%Y-%m-%d %H:%M:%S')[0]]
    if len(df_field) < 50:
        warnings.warn(f'Field measurement count is low. {len(df_field)} usable field visits.')

    # process
    train_x_raw, val_x_raw, test_x_raw, test_y_index, train_df_field, val_df_field, test_df_field = process_tune_data(
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

    # predict water level
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

    # convert to dis
    train_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(train_x_pred_o, 2),
        train_df_field.copy(), df_raw
    )[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(val_x_pred_o, 2),
        val_df_field.copy(), df_raw
    )[:, np.newaxis]

    # apply tuner o1
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    train_df_field, val_df_field = ft.merge_true_wl_dis(train_df_field, val_df_field, df, target_gage)
    base_ratio_o1 = mo.calculate_base_error_ratio(
        train_df_field, val_df_field, data_flood_stage['action'].iloc[0]
    )

    train_x_pred_o_rc, train_x_series, train_x_series_diff = apply_o1_tuner(
        train_x_raw,
        df, target_gage,
        train_x_pred_o, tuner_o1_train_index,
        tuner_o1,
        train_x_pred_o_rc,
        # train_df_field_index,
        base_ratio=base_ratio_o1,
    )
    val_x_pred_o_rc, val_x_series, val_x_series_diff = apply_o1_tuner(
        val_x_raw,
        df, target_gage,
        val_x_pred_o, tuner_o1_val_index,
        tuner_o1,
        val_x_pred_o_rc,
        # val_df_field_index,
        base_ratio=base_ratio_o1,
    )

    # apply tuner o2
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    base_ratio_o2 = mo.calculate_base_error_ratio(
        train_df_field, val_df_field, 0,
    )

    train_x_pred_o_rc = apply_o2_tuner(train_x_pred_o_rc, train_x_pred_o, tuner_o2_list, base_ratio=base_ratio_o2)
    val_x_pred_o_rc = apply_o2_tuner(val_x_pred_o_rc, val_x_pred_o, tuner_o2_list, base_ratio=base_ratio_o2)

    # residual error
    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

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
    test_df_full['field'] = test_df_field['discharge']
    test_df_full['index'] = range(len(test_df_full))
    test_df = test_df_full.loc[~test_df_full['field'].isna(), :].copy()

    test_df_index = test_df['index']
    test_df = test_df.drop('index', axis=1)
    test_df_full = test_df_full.drop('index', axis=1)

    # test set base tune
    test_df = test_df.reset_index()
    pred_error_list = ft.remove_test_base_error(test_df, test_df_full, forward)
    test_df = test_df.set_index('index')
    test_df['pred_water_level_error'] = pred_error_list
    test_df['pred_water_level_u_tuned'] = test_df['pred_water_level']
    test_df['pred_water_level'] = test_df['pred_water_level'] - test_df['pred_water_level_error']

    # convert to dis for the test set
    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o, 2),
        test_df.copy(),
        df_raw
    )[:, np.newaxis]

    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field

    # apply tuner o1 for test set
    residual_pred_o1, test_x_series, test_x_series_diff = apply_o1_tune_4_test(
        test_df, test_x_raw,
        df, target_gage,
        tuner_o1, tuner_o1_test_index,
        test_x_pred_o_rc,
        test_df_index,
    )
    test_x_pred_o_temp = test_x_pred_o_rc.copy()
    test_x_pred_o_temp[tuner_o1_test_index] = (
            test_x_pred_o_temp[tuner_o1_test_index] - residual_pred_o1 * (1 - base_ratio_o1)
    )
    # apply tuner o2 for test set
    residual_pred_o2 = apply_o2_tuner(test_x_pred_o_temp, test_x_pred_o, tuner_o2_list, if_update=False)

    # format
    train_x_full = np.concatenate((train_x_series[:, -1:], train_x_series_diff[:, -1:]), axis=1)
    val_x_full = np.concatenate((val_x_series[:, -1:], val_x_series_diff[:, -1:]), axis=1)
    test_x_full = np.concatenate((test_x_series[:, -1:], test_x_series_diff[:, -1:]), axis=1)

    # calculate base ratio
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    base_ratio_o3 = mo.calculate_base_error_ratio(
        train_df_field, val_df_field, 0
    )

    # residual error learning
    from xgboost import XGBRegressor

    num_sampling = 15
    num_rep = 150

    # norm
    transformer_x_o = PowerTransformer(method='yeo-johnson')
    transformer_x_o.fit(
        np.concatenate((train_x_full, val_x_full), axis=0)
    )

    train_x_full_norm = transformer_x_o.transform(train_x_full)
    val_x_full_norm = transformer_x_o.transform(val_x_full)
    test_x_full_norm = transformer_x_o.transform(test_x_full)

    # train
    residual_pred = np.zeros(test_y_res.shape)
    for i in range(num_rep):
        indices = np.arange(train_x_full_norm.shape[0] + val_x_full_norm.shape[0])
        np.random.shuffle(indices)
        model_res = XGBRegressor(
            n_estimators=500,
            objective='reg:squarederror',
            # early_stopping_rounds=5,
            eval_metric='mae',
            gamma=0,
            # max_depth=max_depth, learning_rate=lr, reg_alpha=reg_alpha,
        )
        model_res.fit(
            np.concatenate((
                train_x_full_norm,
                val_x_full_norm,
            ), axis=0)[indices[:num_sampling]],
            np.concatenate((
                train_y_res,
                val_y_res
            ), axis=0)[indices[:num_sampling]],
            # sample_weight=np.concatenate((
            #     train_y_res_norm[:, 0],
            #     val_y_res_norm[:, 0]
            # ), axis=0)[indices[:num_sampling]],
        )
        residual_pred += model_res.predict(test_x_full_norm)[:, np.newaxis]
    residual_pred = residual_pred / num_rep

    # pred
    pred_y_tune = test_x_pred_o_rc - residual_pred * (1 - base_ratio_o3)
    pred_y_tune[tuner_o1_test_index] = pred_y_tune[tuner_o1_test_index] - residual_pred_o1 * (1 - base_ratio_o1)
    pred_y_tune = pred_y_tune - residual_pred_o2 * (1 - base_ratio_o2)

    # pred only using o1 and o2
    test_x_pred_o_rc_updated = test_x_pred_o_rc.copy()
    test_x_pred_o_rc_updated[tuner_o1_test_index] = (
            test_x_pred_o_rc_updated[tuner_o1_test_index] - residual_pred_o1 * (1 - base_ratio_o1)
    )
    test_x_pred_o_rc_updated = test_x_pred_o_rc_updated - residual_pred_o2 * (1 - base_ratio_o2)

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc_updated
    test_df.loc[:, 'pred'] = pred_y_tune

    # record train val data
    train_df_field['pred_discharge_o2'] = train_x_pred_o_rc
    val_df_field['pred_discharge_o2'] = val_x_pred_o_rc
    train_df_field['pred_discharge_o3'] = train_df_field['pred_discharge_o2'] - model_res.predict(train_x_full_norm)
    val_df_field['pred_discharge_o3'] = val_df_field['pred_discharge_o2'] - model_res.predict(val_x_full_norm)
    train_val_df_field = pd.concat([train_df_field, val_df_field]).sort_index()
    train_val_df_field.to_csv(f'{expr_dir}/train_val_df.csv')

    return test_df, test_df_full
