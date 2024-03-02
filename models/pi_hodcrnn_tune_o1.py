import numpy as np

import torch
import pickle

from models.baselines.hodcrnn_tune_o import process_tune_data

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


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

    # data for tuning: time serie (past values + predicted value) as x
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

    # data for tuning: time serie (past values + predicted value) as x / take off data point long time ago
    train_x_pred_o, train_df_field, train_df_field_index = ft.filter_time_df_field(train_df_field, train_x_pred_o)
    val_x_pred_o, val_df_field, val_df_field_index = ft.filter_time_df_field(val_df_field, val_x_pred_o)

    # data for tuning: time serie (past values + predicted value) as x / shortened model input
    train_x_past = train_x_raw[:, :, 0][train_df_field_index]
    val_x_past = val_x_raw[:, :, 0][val_df_field_index]
    train_x_past = (train_x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                    + df[f"{target_gage}_00065"].min())
    val_x_past = (val_x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                  + df[f"{target_gage}_00065"].min())

    # data for tuning: time serie (past values + predicted value) as x / concat
    train_x_series = np.concatenate((train_x_past, train_x_pred_o), axis=1)
    val_x_series = np.concatenate((val_x_past, val_x_pred_o), axis=1)
    train_x_series_diff = np.diff(train_x_series, axis=1)
    val_x_series_diff = np.diff(val_x_series, axis=1)

    # data for tuning: use residual error as y
    train_x_pred_o = np.round(train_x_pred_o, 2)
    val_x_pred_o = np.round(val_x_pred_o, 2)

    train_x_pred_o_rc = mo.convert_array_w_rc(train_x_pred_o, train_df_field.copy(), df_raw)[:, np.newaxis]
    val_x_pred_o_rc = mo.convert_array_w_rc(val_x_pred_o, val_df_field.copy(), df_raw)[:, np.newaxis]

    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_x_pred_o_rc - train_y_field
    val_y_res = val_x_pred_o_rc - val_y_field

    train_y_res_per = train_y_res / train_x_pred_o_rc
    val_y_res_per = val_y_res / val_x_pred_o_rc

    # test x
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
    test_x_past = test_x_raw[test_df_index.values][:, :, 0]
    test_x_past = (test_x_past * (df[f"{target_gage}_00065"].max() - df[f"{target_gage}_00065"].min())
                   + df[f"{target_gage}_00065"].min())
    test_x_series = np.concatenate((test_x_past, test_x_pred_o), axis=1)
    test_x_series_diff = np.diff(test_x_series, axis=1)

    # test y
    test_x_pred_o_rc = mo.convert_array_w_rc(
        np.round(test_x_pred_o[:, 0], 2),
        test_df.copy(),
        df_raw
    )[:, np.newaxis]

    test_y_field = test_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    test_y_res = test_x_pred_o_rc - test_y_field
    test_y_res_per = test_y_res / test_x_pred_o_rc

    # filter - high flow
    flow_array = np.concatenate((train_x_series[:, -1], val_x_series[:, -1]), axis=0)
    high_flow = (flow_array.mean() + flow_array.std() * 1)  # 6.11
    train_high_flow_index = (train_x_series[:, -2:] >= high_flow).any(axis=1)
    val_high_flow_index = (val_x_series[:, -2:] >= high_flow).any(axis=1)
    test_high_flow_index = (test_x_series[:, -2:] >= high_flow).any(axis=1)

    # filter - high change
    error_rate_array = np.concatenate((
            train_x_series_diff[:, -1] / train_x_series[:, -2],
            val_x_series_diff[:, -1] / val_x_series[:, -2]
    ), axis=0)
    high_change = (error_rate_array.mean() + error_rate_array.std() * 1)  # 0.015
    train_high_change_index = np.abs(train_x_series_diff[:, -1] / train_x_series[:, -2]) >= high_change
    val_high_change_index = np.abs(val_x_series_diff[:, -1] / val_x_series[:, -2]) >= high_change
    test_high_change_index = np.abs(test_x_series_diff[:, -1] / test_x_series[:, -2]) >= high_change

    # filter - all
    train_filter = train_high_flow_index & train_high_change_index
    val_filter = val_high_flow_index & val_high_change_index
    test_filter = test_high_flow_index & test_high_change_index

    # filtering
    train_x_series_diff_tune = train_x_series_diff[:, -1][train_filter]
    train_x_series_tune = train_x_series[:, -2][train_filter]
    val_x_series_diff_tune = val_x_series_diff[:, -1][val_filter]
    val_x_series_tune = val_x_series[:, -2][val_filter]
    test_x_series_diff_tune = test_x_series_diff[:, -1][test_filter]
    test_x_series_tune = test_x_series[:, -2][test_filter]

    # vis
    plt.scatter(
        np.concatenate((
            train_x_series_diff_tune / train_x_series_tune,
            val_x_series_diff_tune / val_x_series_tune,
        ), axis=0),
        np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0),
        color = 'red'
    )
    plt.scatter(
        test_x_series_diff_tune / test_x_series_tune,
        test_y_res_per[test_filter],
        color = 'blue'
    )
    plt.show()

    # residual error learning
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    degree = 1
    model_res = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_res.fit(
        np.concatenate((
            (train_x_series_diff_tune / train_x_series_tune)[:, np.newaxis],
            (val_x_series_diff_tune / val_x_series_tune)[:, np.newaxis],
        ), axis=0),
        np.concatenate((train_y_res_per[train_filter], val_y_res_per[val_filter]), axis=0),
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

    pred_y_tune = test_x_pred_o_rc
    pred_y_tune[test_filter] = pred_y_tune[test_filter] - residual_pred

    # recording
    test_df_full.loc[:, 'pred'] = np.nan
    test_df.loc[:, 'pred_w_o_tune'] = test_x_pred_o_rc
    test_df.loc[:, 'pred'] = pred_y_tune

    return test_df, test_df_full
