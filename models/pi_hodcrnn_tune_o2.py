import numpy as np
import pandas as pd

import torch
import pickle

from utils.features import process_tune_data

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
import os
import re


def apply_o1_tuner(
        x_raw,
        df, target_gage,
        x_pred_o, tuner_o1_index,
        tuner_o1,
        x_pred_o_rc,
        df_field_filter_index=None,
        base_ratio=0,
):
    if df_field_filter_index is not None:
        x_past = x_raw[:, :, 0][df_field_filter_index]
    else:
        x_past = x_raw[:, :, 0]
    x_past = (x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                    + df[f"{target_gage}_00065"].min())

    x_series = np.concatenate((x_past, x_pred_o), axis=1)
    x_series_diff = np.diff(x_series, axis=1)

    x_series_diff_tune = x_series_diff[:, -1][tuner_o1_index]
    x_series_tune = x_series[:, -2][tuner_o1_index]

    if (x_series_diff_tune / x_series_tune).size == 0:
        x_pred_o_rc_o1 = x_pred_o_rc.copy()
    else:
        residual_pred = tuner_o1.predict((x_series_diff_tune / x_series_tune)[:, np.newaxis])
        residual_pred = residual_pred * x_pred_o_rc[tuner_o1_index]

        x_pred_o_rc_o1 = x_pred_o_rc.copy()
        x_pred_o_rc_o1[tuner_o1_index] = x_pred_o_rc[tuner_o1_index] - residual_pred * (1 - base_ratio)

    return x_pred_o_rc_o1, x_series, x_series_diff


def apply_o1_tune_4_test(
        test_df, test_x_raw,
        df, target_gage,
        tuner_o1, tuner_o1_test_index,
        test_x_pred_o_rc,
        test_df_index,
):
    # apply tuner o1
    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_past = test_x_raw[test_df_index.values][:, :, 0]
    test_x_past = (test_x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                   + df[f"{target_gage}_00065"].min())
    test_x_series = np.concatenate((test_x_past, test_x_pred_o), axis=1)
    test_x_series_diff = np.diff(test_x_series, axis=1)

    test_x_series_diff_tune = test_x_series_diff[:, -1][tuner_o1_test_index]
    test_x_series_tune = test_x_series[:, -2][tuner_o1_test_index]

    residual_pred_o1 = tuner_o1.predict((test_x_series_diff_tune / test_x_series_tune)[:, np.newaxis])
    residual_pred_o1 = residual_pred_o1 * test_x_pred_o_rc[tuner_o1_test_index]
    return residual_pred_o1, test_x_series, test_x_series_diff


def check_bias_trend(train_df_field, val_df_field, train_x_pred_o_rc, val_x_pred_o_rc):
    train_df_field['dis_pred_rc'] = train_x_pred_o_rc
    train_df_field['error'] = train_df_field['dis_pred_rc'] - train_df_field['discharge']
    train_df_field['%error'] = train_df_field['error'] / train_df_field['discharge']
    val_df_field['dis_pred_rc'] = val_x_pred_o_rc
    val_df_field['error'] = val_df_field['dis_pred_rc'] - val_df_field['discharge']
    val_df_field['%error'] = val_df_field['error'] / val_df_field['discharge']
    train_val_df_field = pd.concat((train_df_field, val_df_field), axis=0)
    train_val_df_field = train_val_df_field.sort_index()
    train_val_df_field['%error_cum'] = train_val_df_field['%error'].cumsum()
    train_val_df_field['%error_cum'].plot()
    return


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, data_flood_stage, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model.name = 'LevelPredHomoDCRNN_tune'

    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_1__2024-04-04-13-15-56'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_2__2024-04-04-13-26-18'
    dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_3__2024-04-04-13-27-54'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_4__2024-04-04-13-29-34'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_5__2024-04-04-13-30-53'
    # dir_o1 = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_6__2024-04-04-13-32-38'
    with open(f'{dir_o1}/tuner_o1_1degree_poly.pkl', 'rb') as file:
        tuner_o1 = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_train.pkl', 'rb') as file:
        tuner_o1_train_index = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_val.pkl', 'rb') as file:
        tuner_o1_val_index = pickle.load(file)
    with open(f'{dir_o1}/tuner_o1_apply_index_test.pkl', 'rb') as file:
        tuner_o1_test_index = pickle.load(file)

    # tuner params
    model_res_name = 'local_reg'
    scaler = MinMaxScaler # StandardScaler

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
    start_time = df_normed[df_normed['index'] == sequences_w_index[0, 0, -1]].index
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

    # convert to dis
    train_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(train_x_pred_o, 2),
        train_df_field.copy(), df_raw, target_gage
    )[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(val_x_pred_o, 2),
        val_df_field.copy(), df_raw, target_gage
    )[:, np.newaxis]

    # calculate base error ratio
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    train_df_field, val_df_field = ft.merge_true_wl_dis(train_df_field, val_df_field, df, target_gage)
    base_ratio_o1 = mo.calculate_base_error_ratio(
        train_df_field, val_df_field, data_flood_stage['action'].iloc[0]
    )

    # apply o1
    train_x_pred_o_rc, _, _ = apply_o1_tuner(
        train_x_raw,
        df, target_gage,
        train_x_pred_o, tuner_o1_train_index,
        tuner_o1,
        train_x_pred_o_rc,
        # train_df_field_index,
        base_ratio=base_ratio_o1,
    )
    val_x_pred_o_rc, _, _ = apply_o1_tuner(
        val_x_raw,
        df, target_gage,
        val_x_pred_o, tuner_o1_val_index,
        tuner_o1,
        val_x_pred_o_rc,
        # val_df_field_index,
        base_ratio=base_ratio_o1,
    )

    # record data
    train_df_field['pred_discharge'] = train_x_pred_o_rc
    val_df_field['pred_discharge'] = val_x_pred_o_rc
    train_val_df_field = pd.concat([train_df_field, val_df_field]).sort_index()
    train_val_df_field.to_csv(f'{expr_dir}/train_val_df.csv')

    # calculate base error ratio
    base_ratio_o2 = mo.calculate_base_error_ratio(train_df_field, val_df_field, 0)

    # residual error
    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

    train_y_res = train_y_res / train_x_pred_o_rc
    val_y_res = val_y_res / val_x_pred_o_rc

    train_y_res_w_weight = np.concatenate((
        train_df_field['discharge_weights'].values[:, np.newaxis], train_y_res,
    ), axis=1)
    val_y_res_w_weight = np.concatenate((
        val_df_field['discharge_weights'].values[:, np.newaxis], val_y_res,
    ), axis=1)

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

    # convert for test set
    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o, 2),
        test_df.copy(),
        df_raw, target_gage
    )[:, np.newaxis]

    # apply o1 for test set
    residual_pred_o1, _, _ = apply_o1_tune_4_test(
        test_df, test_x_raw,
        df, target_gage,
        tuner_o1, tuner_o1_test_index,
        test_x_pred_o_rc,
        test_df_index,
    )
    test_x_pred_o_rc[tuner_o1_test_index] = (
            test_x_pred_o_rc[tuner_o1_test_index]
            - residual_pred_o1 * (1 - base_ratio_o1)
    )

    # residual error for test set
    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field
    test_y_res = test_y_res / test_x_pred_o_rc

    # checking for take off data point long time ago, when measurement error might be more dominant
    # only open when developing
    # check_bias_trend(train_df_field, val_df_field, train_x_pred_o_rc, val_x_pred_o_rc)

    # residual error learning
    if model_res_name == 'xgboost':
        from models.baselines.xgboost import XGBRegressor

        num_sampling = 59
        num_rep = 40

        # norm
        transformer_x_o = PowerTransformer(method='yeo-johnson')
        transformer_x_o.fit(df[f"{target_gage}_00060"].values[:len(df) // 2, np.newaxis])

        train_x_pred_o_norm = transformer_x_o.transform(train_x_pred_o)
        val_x_pred_o_norm = transformer_x_o.transform(val_x_pred_o)
        test_x_pred_o_norm = transformer_x_o.transform(test_x_pred_o)

        transformer_y = PowerTransformer(method='yeo-johnson')
        transformer_y.fit(np.concatenate(
            (train_y_res, val_y_res),
            axis=0)
        )
        train_y_res_norm = np.concatenate((
            train_df_field['discharge_weights'].values[:, np.newaxis],
            transformer_y.transform(train_y_res)),
            axis=1,
        )
        val_y_res_norm = np.concatenate((
            val_df_field['discharge_weights'].values[:, np.newaxis],
            transformer_y.transform(val_y_res)),
            axis=1,
        )

        # train
        residual_pred = np.zeros(test_x_pred_o_norm.shape)
        for i in range(num_rep):
            indices = np.arange(train_x_pred_o_norm.shape[0] + val_x_pred_o_norm.shape[0])
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
                    train_x_pred_o_norm,
                    val_x_pred_o_norm,
                ), axis=0)[indices[:num_sampling]],
                np.concatenate((
                    train_y_res_norm[:, 1:],
                    val_y_res_norm[:, 1:]
                ), axis=0)[indices[:num_sampling]],
                # eval_set = [
                #     (train_x_tune, train_y_tune[:, 1:]),
                #     (val_x_tune, val_y_tune[:, 1:])
                # ],
                sample_weight=np.concatenate((
                    train_y_res_norm[:, 0],
                    val_y_res_norm[:, 0]
                ), axis=0)[indices[:num_sampling]],
            )
            residual_pred += model_res.predict(test_x_pred_o_norm)[:, np.newaxis]
        residual_pred = residual_pred / num_rep
        residual_pred = transformer_y.inverse_transform(residual_pred)
        residual_pred = residual_pred * test_x_pred_o_rc

    if model_res_name == 'poly':
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        y_weights = pp.sample_weights(
            pd.DataFrame(np.concatenate((train_y_res, val_y_res), axis=0), columns=['%error']),
            '%error', if_log=True
        )['%error_weights'].values

        degree = 2
        model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_res.fit(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            # linearregression__sample_weight= (
            #         y_weights ** 2
            #         # + np.concatenate((train_y_res_w_weight[:, 0], val_y_res_w_weight[:, 0]), axis=0)
            # )
        )
        residual_pred = model_res.predict(test_x_pred_o)
        residual_pred = residual_pred * test_x_pred_o_rc

        # save and vis
        with open(f'{expr_dir}/tuner_o2_poly_{degree}_order.pkl', 'wb') as f:
            pickle.dump(model_res, f)

        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train')
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test')
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 100).reshape(-1, 1)
        y_plot = model_res.predict(x_plot)
        plt.plot(x_plot, y_plot, color='blue', label=f'Polynomial degree {degree}')
        plt.show()

    if model_res_name == 'norm_poly':
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        # norm
        transformer_x_o = PowerTransformer(method='yeo-johnson')
        transformer_x_o.fit(df[f"{target_gage}_00060"].values[:len(df) // 2, np.newaxis])

        train_x_pred_o_norm = transformer_x_o.transform(train_x_pred_o)
        val_x_pred_o_norm = transformer_x_o.transform(val_x_pred_o)
        test_x_pred_o_norm = transformer_x_o.transform(test_x_pred_o)

        transformer_y = PowerTransformer(method='yeo-johnson')
        transformer_y.fit(np.concatenate(
            (train_y_res, val_y_res),
            axis=0)
        )
        train_y_res_norm = np.concatenate((
            train_df_field['discharge_weights'].values[:, np.newaxis],
            transformer_y.transform(train_y_res)),
            axis=1,
        )
        val_y_res_norm = np.concatenate((
            val_df_field['discharge_weights'].values[:, np.newaxis],
            transformer_y.transform(val_y_res)),
            axis=1,
        )

        # train
        degree = 3
        model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_res.fit(
            np.concatenate((train_x_pred_o_norm, val_x_pred_o_norm), axis=0),
            np.concatenate((train_y_res_norm[:, 1:], val_y_res_norm[:, 1:]), axis=0),
            # linearregression__sample_weight=np.concatenate((train_y_res_w_weight[:, 0], val_y_res_w_weight[:, 0]), axis=0)
        )
        residual_pred = model_res.predict(test_x_pred_o_norm)
        residual_pred = transformer_y.inverse_transform(residual_pred)
        residual_pred = residual_pred * test_x_pred_o_rc

        # save and vis
        with open(f'{expr_dir}/tuner_o2_norm_poly_{degree}_order.pkl', 'wb') as f:
            pickle.dump(model_res, f)

        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 100).reshape(-1, 1)
        y_plot = transformer_y.inverse_transform(
            model_res.predict(transformer_x_o.transform(x_plot))
        )
        plt.plot(x_plot, y_plot, color='blue', label=f'Polynomial degree {degree}')
        plt.show()

    if model_res_name == 'log_poly':
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        # log
        train_x_pred_o_log = np.log(train_x_pred_o)
        val_x_pred_o_log = np.log(val_x_pred_o)
        test_x_pred_o_log = np.log(test_x_pred_o)

        y_weights = pp.sample_weights(
            pd.DataFrame(np.concatenate((train_y_res, val_y_res), axis=0), columns=['%error']),
            '%error', if_log=True
        )['%error_weights'].values

        # train
        degree = 2
        model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_res.fit(
            np.concatenate((train_x_pred_o_log, val_x_pred_o_log), axis=0),
            np.concatenate((train_y_res, val_y_res), axis=0),
            linearregression__sample_weight= (
                y_weights ** 2 +
                np.concatenate((train_y_res_w_weight[:, 0], val_y_res_w_weight[:, 0]), axis=0) ** 2
            )
        )
        residual_pred = model_res.predict(test_x_pred_o_log)
        residual_pred = residual_pred * test_x_pred_o_rc

        # save and vis
        with open(f'{expr_dir}/tuner_o2_log_poly_{degree}_order.pkl', 'wb') as f:
            pickle.dump(model_res, f)

        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 100).reshape(-1, 1)
        y_plot = model_res.predict(np.log(x_plot))

        plt.plot(x_plot, y_plot, color='blue', label=f'Polynomial degree {degree}')
        plt.show()

    if model_res_name == 'local_reg':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        from scipy.interpolate import interp1d

        # train
        frac = 0.7
        smoothed = lowess(
            np.concatenate((train_y_res, val_y_res), axis=0)[:, 0],
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0)[:, 0],
            frac=frac
        )

        # interpolate
        interpolation_function = interp1d(smoothed[:, 0], smoothed[:, 1])
        residual_pred = np.full(test_x_pred_o.shape, np.nan)
        index_within_range = (test_x_pred_o >= smoothed[:, 0].min()) & (test_x_pred_o <= smoothed[:, 0].max())
        residual_pred[index_within_range] = interpolation_function(test_x_pred_o[index_within_range])

        residual_pred_t = np.full(test_x_pred_o[~index_within_range].shape, np.nan)
        i = 0
        for v in test_x_pred_o[~index_within_range]:
            differences = np.abs(smoothed[:, 0] - v)
            closest_x = smoothed[:, 0][np.argpartition(differences, 2)[:2]]
            closest_y = smoothed[:, 1][np.argpartition(differences, 2)[:2]]
            v_y = ((v - closest_x[1]) * (closest_y[0] - closest_y[1]) / (closest_x[0] - closest_x[1])) + closest_y[1]
            residual_pred_t[i] = v_y
        residual_pred[~index_within_range] = residual_pred_t

        residual_pred = residual_pred * test_x_pred_o_rc

        with open(f'{expr_dir}/tuner_o2_lowess.pkl', 'wb') as f:
            pickle.dump(smoothed, f)

        # vis
        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='blue',)
        plt.show()

    if model_res_name == 'gam':
        from pygam import s, GAM

        y_weights = pp.sample_weights(
            pd.DataFrame(np.concatenate((train_y_res, val_y_res), axis=0), columns=['%error']),
            '%error', if_log=True
        )['%error_weights'].values

        model_res = GAM(s(0,))
        model_res.fit(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res, val_y_res), axis=0),
            # weights=y_weights
        )

        residual_pred = model_res.predict(test_x_pred_o)[:, np.newaxis]
        residual_pred = residual_pred * test_x_pred_o_rc

        # # save and vis
        with open(f'{expr_dir}/tuner_o2_gam.pkl', 'wb') as f:
            pickle.dump(model_res, f)

        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 100).reshape(-1, 1)
        y_plot = model_res.predict(x_plot)
        plt.plot(x_plot, y_plot, color='blue')
        plt.show()

    if model_res_name == 'seg_local_reg':
        import ruptures as rpt
        from statsmodels.nonparametric.smoothers_lowess import lowess
        from scipy.interpolate import interp1d

        # get breakpoint
        x = np.concatenate((train_x_pred_o, val_x_pred_o), axis=0)
        y = np.concatenate((train_y_res, val_y_res), axis=0)
        indices = np.argsort(x, axis=0)
        x_sorted = x[indices[:, 0]]
        y_sorted = y[indices[:, 0]]
        result = rpt.Dynp(model='l1').fit(y_sorted).predict(n_bkps=1)
        # rpt.display(y, result, figsize=(10, 6))
        # plt.show()

        # manual
        result = [39, 64]

        # reg
        i = 0
        residual_pred = np.full(test_x_pred_o.shape, np.nan)
        x_brk_list = []
        model_res_list = []
        frac_list = [1, 1]
        for seg_begin, seg_end in zip([0] + result[:-1], result):

            frac = frac_list[i]
            model_res = lowess(
                y_sorted[seg_begin: seg_end][:, 0],
                x_sorted[seg_begin: seg_end][:, 0],
                frac=frac
            )

            model_res_list.append(model_res)
            with open(f'{expr_dir}/tuner_o2_lowess_seg{i}.pkl', 'wb') as f:
                pickle.dump(model_res, f)
            i += 1

            x_threshold_up = x_sorted[seg_begin: seg_end].max()
            x_threshold_low = x_sorted[seg_begin: seg_end].min()
            x_brk_list.append([x_threshold_low, x_threshold_up])
            test_x_index = (test_x_pred_o <= x_threshold_up) & (test_x_pred_o >= x_threshold_low)

            interpolation_function = interp1d(model_res[:, 0], model_res[:, 1])
            residual_pred[test_x_index] = interpolation_function(test_x_pred_o[test_x_index])

        assert ~np.isnan(residual_pred).any(), 'Missing prediction.'
        residual_pred = residual_pred * test_x_pred_o_rc

        # vis
        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 1000).reshape(-1, 1)
        for x_brk, model_res in zip(x_brk_list, model_res_list):
            x_plot_select = x_plot[(x_plot >= model_res[:, 0].min()) & (x_plot <= model_res[:, 0].max())]

            interpolation_function = interp1d(model_res[:, 0], model_res[:, 1])
            y_plot = interpolation_function(x_plot_select)

            plt.plot(x_plot_select, y_plot, color='blue')
        plt.show()

    if model_res_name == 'seg_poly':
        import ruptures as rpt
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        y_weights = pp.sample_weights(
            pd.DataFrame(np.concatenate((train_y_res, val_y_res), axis=0), columns=['%error']),
            '%error', if_log=True
        )['%error_weights'].values

        # get breakpoint
        x = np.concatenate((train_x_pred_o, val_x_pred_o), axis=0)
        y = np.concatenate((train_y_res, val_y_res), axis=0)
        indices = np.argsort(x, axis=0)
        x_sorted = x[indices[:, 0]]
        y_sorted = y[indices[:, 0]]
        result = rpt.Dynp(model='l1').fit(y_sorted).predict(n_bkps=1)
        # rpt.display(y, result, figsize=(10, 6))
        # plt.show()

        # manual
        result = [39, 64]

        # reg
        i = 0
        residual_pred = np.full(test_x_pred_o.shape, np.nan)
        x_brk_list = []
        model_res_list = []
        for seg_begin, seg_end in zip([0] + result[:-1], result):

            degree = 2
            model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model_res.fit(
                x_sorted[seg_begin: seg_end],
                y_sorted[seg_begin: seg_end],
                # linearregression__sample_weight= (
                #         y_weights ** 2
                    # + np.concatenate((train_y_res_w_weight[:, 0], val_y_res_w_weight[:, 0]), axis=0)
            # )
            )

            model_res_list.append(model_res)
            with open(f'{expr_dir}/tuner_o2_poly_seg{i}.pkl', 'wb') as f:
                pickle.dump(model_res, f)
            i += 1

            x_threshold_up = x_sorted[seg_begin: seg_end].max()
            x_threshold_low = x_sorted[seg_begin: seg_end].min()
            x_brk_list.append([x_threshold_low, x_threshold_up])
            test_x_index = (test_x_pred_o <= x_threshold_up) & (test_x_pred_o >= x_threshold_low)

            residual_pred[test_x_index] = model_res.predict(test_x_pred_o[test_x_index][:, np.newaxis])[:, 0]

        assert ~np.isnan(residual_pred).any(), 'Missing prediction.'
        residual_pred = residual_pred * test_x_pred_o_rc

        # vis
        plt.scatter(
            np.concatenate((train_x_pred_o, val_x_pred_o), axis=0),
            np.concatenate((train_y_res_w_weight[:, 1:], val_y_res_w_weight[:, 1:]), axis=0),
            color='red', label='train', s=5)
        plt.scatter(
            test_x_pred_o,
            test_y_res,
            color='blue', label='test', s=5)
        x_plot = np.linspace(train_x_pred_o.min(), train_x_pred_o.max(), 1000).reshape(-1, 1)
        for x_brk, model_res in zip(x_brk_list, model_res_list):
            x_plot_select = x_plot[(x_plot >= x_brk[0]) & (x_plot <= x_brk[1])]
            y_plot = model_res.predict(x_plot_select[:, np.newaxis])

            plt.plot(x_plot_select, y_plot, color='blue')
        plt.show()

    pred_y_tune = test_x_pred_o_rc - residual_pred * (1 - base_ratio_o2)

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc
    test_df.loc[:, 'pred'] = pred_y_tune

    return test_df, test_df_full
