import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp
from utils.features import process_tune_data

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = MinMaxScaler # StandardScaler

    # reload model
    saved_model = torch.load(
        './outputs/experiments/ARCHIVE_pi_hodcrnn_1__2024-03-14-17-37-25/best_HODCRNN_optuna_tune_0.00023879233049228787.pth'
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
    area_ratio_precip = pd.read_csv(f'{adj_matrix_dir}/area_in_boundary_ratio.csv')
    area_ratio_precip['lat'] = area_ratio_precip['identifier'].str.split('_').str.get(0)
    area_ratio_precip['lat'] = area_ratio_precip['lat'].astype(float)
    area_ratio_precip['lat'] = area_ratio_precip['lat'] - 0.05
    area_ratio_precip['lon'] = area_ratio_precip['identifier'].str.split('_').str.get(1)
    area_ratio_precip['lon'] = area_ratio_precip['lon'].astype(float)
    area_ratio_precip['lon'] = area_ratio_precip['lon'] - 0.05
    area_ratio_precip['label'] = area_ratio_precip.apply(
        lambda x: f"clat{round(x['lat'], 1)}_clon{round(x['lon'], 1)}",
        axis=1,
    )
    df_precip_scaled = df_precip[area_ratio_precip['label'].to_list()]
    for col in df_precip_scaled.columns:
        df_precip_scaled.loc[:, col] = df_precip_scaled[col] * area_ratio_precip[
            area_ratio_precip['label'] == col
            ]['updated_area_ratio'].iloc[0]
    df_precip_scaled = df_precip_scaled.sum(axis=1).to_frame()
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

    # keep usable field measurements (new)
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

    # x / predicted water level
    train_x_pred_o = model(
        torch.tensor(train_x_raw).to(device, dtype=torch.float)
    ).detach().cpu().numpy()[:, 0:1, 0].astype('float64')
    val_x_pred_o = model(
        torch.tensor(val_x_raw).to(device, dtype=torch.float)
    ).detach().cpu().numpy()[:, 0:1, 0].astype('float64')

    scaler_pred = scaler()
    scaler_pred.fit(df[[f"{target_gage}_00065"]])
    train_x_pred_o = scaler_pred.inverse_transform(pd.DataFrame(train_x_pred_o))
    val_x_pred_o = scaler_pred.inverse_transform(pd.DataFrame(val_x_pred_o))

    # x / take off data point long time ago
    train_x_pred_o, train_df_field, train_df_field_index = ft.filter_time_df_field(train_df_field, train_x_pred_o)
    val_x_pred_o, val_df_field, val_df_field_index = ft.filter_time_df_field(val_df_field, val_x_pred_o)

    # x / get last time point value
    train_x_last = train_x_raw[:, -1:, 0][train_df_field_index]
    train_x_last = scaler_pred.inverse_transform(pd.DataFrame(train_x_last))
    val_x_last = val_x_raw[:, -1:, 0][val_df_field_index]
    val_x_last = scaler_pred.inverse_transform(pd.DataFrame(val_x_last))

    # y / predicted discharge
    train_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(train_x_pred_o, 2),
        train_df_field.copy(), df_raw
    )[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(val_x_pred_o, 2),
        val_df_field.copy(), df_raw
    )[:, np.newaxis]

    # y / error rate
    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

    # test set
    test_x_last = test_x_raw[:, -1:, 0]
    test_x_last = scaler_pred.inverse_transform(pd.DataFrame(test_x_last))

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

    test_df_index = test_df['index'].values
    test_df = test_df.drop('index', axis=1)
    test_df_full = test_df_full.drop('index', axis=1)

    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o, 2),
        test_df.copy(),
        df_raw
    )[:, np.newaxis]

    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field

    # reorganize x
    train_x_full = np.concatenate((train_x_pred_o, train_x_pred_o - train_x_last), axis=1)
    val_x_full = np.concatenate((val_x_pred_o, val_x_pred_o - val_x_last), axis=1)
    test_x_full = np.concatenate((test_x_pred_o, test_x_pred_o - test_x_last[test_df_index]), axis=1)

    # residual error learning
    from models.baselines.xgboost import XGBRegressor

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
    pred_y_tune = test_x_pred_o_rc - residual_pred

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc
    test_df.loc[:, 'pred'] = pred_y_tune

    return test_df, test_df_full
