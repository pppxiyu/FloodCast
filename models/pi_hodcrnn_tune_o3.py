import pickle

import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp
from models.baselines.hodcrnn_tune_o import process_tune_data
from models.pi_hodcrnn_tune_o2 import apply_o1_tuner, apply_o1_tune_4_test
import scipy.interpolate as interpolate


def apply_o2_tuner(x_pred_o_rc, x_pred_o, tuner_o2_list, if_update=True):
    residual_pred = np.full(x_pred_o_rc.shape, np.nan)
    for tuner_o2 in tuner_o2_list:
        interpolation_function = interpolate.interp1d(tuner_o2[:, 0], tuner_o2[:, 1])
        apply_index = (x_pred_o <= tuner_o2[:, 0].max()) & (x_pred_o >= tuner_o2[:, 0].min())
        residual_pred[apply_index] = interpolation_function(x_pred_o[apply_index])
    assert ~np.isnan(residual_pred).any(), 'Missing prediction.'
    residual_pred = residual_pred * x_pred_o_rc
    if if_update:
        train_x_pred_o_rc_updated = x_pred_o_rc - residual_pred
        return train_x_pred_o_rc_updated
    else:
        return residual_pred


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reload model
    saved_model = torch.load('./outputs/USGS_01573560/best_level_HODCRNN_optuna_tune_0.000359405908966437.pth')
    model = saved_model['model']
    model.eval()
    model.to(device)
    model.name = 'LevelPredHomoDCRNN_tune'

    with open('./outputs/USGS_01573560/tuner_o1_1degree_poly.pkl', 'rb') as file:
        tuner_o1 = pickle.load(file)
    with open('./outputs/USGS_01573560/tuner_o1_apply_index_train.pkl', 'rb') as file:
        tuner_o1_train_index = pickle.load(file)
    with open('./outputs/USGS_01573560/tuner_o1_apply_index_val.pkl', 'rb') as file:
        tuner_o1_val_index = pickle.load(file)
    with open('./outputs/USGS_01573560/tuner_o1_apply_index_test.pkl', 'rb') as file:
        tuner_o1_test_index = pickle.load(file)

    tuner_o2_list = []
    with open('./outputs/USGS_01573560/tuner_o2_lowess_seg0.pkl', 'rb') as file:
        tuner_o2_list.append(pickle.load(file))
    with open('./outputs/USGS_01573560/tuner_o2_lowess_seg1.pkl', 'rb') as file:
        tuner_o2_list.append(pickle.load(file))

    # tuner params
    # model_res_name = 'seg_local_reg'

    # data
    df_raw = df.copy()
    df = df.resample('H', closed='right', label='right').mean()
    df_wl_normed = (df - df.min()) / (df.max() - df.min())
    wl_cols = [col for col in df.columns if col.endswith('00065')]
    df_wl_normed = df_wl_normed[wl_cols]
    for col in df_wl_normed:
        if col.endswith('00065'):
            df_wl_normed = pp.sample_weights(df_wl_normed, col, if_log=True)

    # inputs
    target_in_forward = 1
    inputs = (
            sorted([col for col in df_wl_normed if "_weights" in col], reverse=True)
            + sorted([col for col in wl_cols if "_weights" not in col], reverse=True)
    )

    # make sequences and remove samples with nan values
    df_wl_normed['index'] = range(len(df_wl_normed))
    sequences_w_index = ft.create_sequences(df_wl_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # process
    train_x_raw, val_x_raw, test_x_raw, test_y_index, train_df_field, val_df_field, test_df_field = process_tune_data(
        df_field, df_wl_normed,
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

    train_x_pred_o = (train_x_pred_o * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                    + df[f"{target_gage}_00065"].min())
    val_x_pred_o = (val_x_pred_o * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                    + df[f"{target_gage}_00065"].min())

    # x / take off data point long time ago
    train_x_pred_o, train_df_field, train_df_field_index = ft.filter_time_df_field(train_df_field, train_x_pred_o)
    val_x_pred_o, val_df_field, val_df_field_index = ft.filter_time_df_field(val_df_field, val_x_pred_o)

    # y / predicted discharge
    train_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(train_x_pred_o, 2),
        train_df_field.copy(), df_raw
    )[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(val_x_pred_o, 2),
        val_df_field.copy(), df_raw
    )[:, np.newaxis]

    # y / apply tuner o1
    train_x_pred_o_rc, train_x_series, train_x_series_diff = apply_o1_tuner(
        train_x_raw,
        df, target_gage,
        train_x_pred_o, tuner_o1_train_index,
        tuner_o1,
        train_x_pred_o_rc,
        train_df_field_index,
    )
    val_x_pred_o_rc, val_x_series, val_x_series_diff = apply_o1_tuner(
        val_x_raw,
        df, target_gage,
        val_x_pred_o, tuner_o1_val_index,
        tuner_o1,
        val_x_pred_o_rc,
        val_df_field_index,
    )

    # y / apply tuner o2
    train_x_pred_o_rc = apply_o2_tuner(train_x_pred_o_rc, train_x_pred_o, tuner_o2_list)
    val_x_pred_o_rc = apply_o2_tuner(val_x_pred_o_rc, val_x_pred_o, tuner_o2_list)

    # y / error rate
    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

    # test set
    test_x_pred_o = mo.pred_4_test_hodcrnn(model, test_x_raw, target_in_forward, device)
    test_x_pred_o = test_x_pred_o[:, 0, :].astype(np.float64)
    test_x_pred_o = (test_x_pred_o * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                   + df[f"{target_gage}_00065"].min())

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

    test_x_pred_o = test_df['pred_water_level'].values[:, np.newaxis]
    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o, 2),
        test_df.copy(),
        df_raw
    )[:, np.newaxis]

    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field

    # apply tuner o1
    residual_pred_o1, test_x_series, test_x_series_diff = apply_o1_tune_4_test(
        test_df, test_x_raw,
        df, target_gage,
        tuner_o1, tuner_o1_test_index,
        test_x_pred_o_rc,
        test_df_index,
    )

    # apply tuner o2
    residual_pred_o2 = apply_o2_tuner(test_x_pred_o_rc, test_x_pred_o, tuner_o2_list, if_update=False)

    # reorganize x
    train_x_full = np.concatenate((train_x_series[:, -1:], train_x_series_diff[:, -1:]), axis=1)
    val_x_full = np.concatenate((val_x_series[:, -1:], val_x_series_diff[:, -1:]), axis=1)
    test_x_full = np.concatenate((test_x_series[:, -1:], test_x_series_diff[:, -1:]), axis=1)

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
    pred_y_tune[tuner_o1_test_index] = pred_y_tune[tuner_o1_test_index] - residual_pred_o1
    pred_y_tune = pred_y_tune - residual_pred_o2

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc
    test_df.loc[:, 'pred'] = pred_y_tune

    return test_df, test_df_full
