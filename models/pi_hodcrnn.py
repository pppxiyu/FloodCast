import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import optuna

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from models.hodcrnn import HomoDCRNN

import warnings
import json

from sklearn.preprocessing import MinMaxScaler


expr_dir_global = ''
best_score_optuna_tune = 1e10


def rc_checker(df, dict_rc, df_field):

    # get rc
    max_level_target = df['01573560_00065'].max()
    max_dis_target = df['01573560_00060'].max()
    df_rc = pd.DataFrame.from_dict(dict_rc, orient='index', columns=['dis'])
    df_rc = df_rc[df_rc.index <= max_level_target + 2]
    df_rc = df_rc[df_rc['dis'] <= max_dis_target + 10000]

    original_index = df_rc.index
    df_rc.index = [f"{x:.4f}" for x in df_rc.index]
    df_rc = df_rc.reindex(pd.Index(
        [f"{x:.4f}" for x in np.arange(original_index.min(), original_index.max() + 0.0001, 0.0001)]
    ))
    df_rc.index = df_rc.index.astype(float).round(4)
    df_rc = df_rc.interpolate(method='linear', order=3, axis=0)

    # rc check
    df_curve_check = df[['01573560_00065', '01573560_00060']]
    df_curve_check.loc[:, '01573560_00065'] = df_curve_check['01573560_00065'].round(4)
    df_curve_check = df_curve_check.reset_index()
    df_curve_check = df_curve_check.merge(df_rc, how='left', left_on='01573560_00065', right_on=df_rc.index)
    df_curve_check['new_rc_error'] = df_curve_check['01573560_00060'] - df_curve_check['dis']

    # approx rc check
    df_field_test = df_field.copy()
    df_field_test.index = df_field_test.index.ceil('H')

    df_approx_rc = df[['01573560_00065', '01573560_00060']]
    df_approx_rc_sampled = df_approx_rc.loc[df_field_test.index].reset_index()

    df_approx_rc_sampled['approx_rc_dis'] = df_approx_rc_sampled.apply(
        lambda row: mo.approx_rc(row, df_approx_rc), axis=1
    )
    df_approx_rc_sampled['%_bias'] = (
            df_approx_rc_sampled['01573560_00060'] - df_approx_rc_sampled['approx_rc_dis']
    ) / df_approx_rc_sampled['01573560_00060'] * 100
    print(df_approx_rc_sampled['%_bias'].mean())
    return


def get_rc_shifts():
    rc_shift_records = pd.read_csv(
        './data/USGS_gage_01573560_shift_curves/01573560_shift_records.csv', index_col=False
    )
    rc_shift_records['Start Time'] = pd.to_datetime(rc_shift_records['Start Time'])
    rc_shift_records['Start Time'] = rc_shift_records['Start Time'].dt.tz_localize('America/New_York')
    rc_shift_records['Shift number'] = rc_shift_records.apply(
        lambda r: r['Shift number'] if r['End Time'] == 'Open' else 'base',
        axis=1
    )
    rc_shift_records['End Time'] = rc_shift_records['Start Time'].shift(-1)

    rc_shift_dict = {}
    for index, row in rc_shift_records.iterrows():
        curve_num = row['Curve Number']
        shift_num = row['Shift number']
        rc_shift_dict[(curve_num, shift_num)] = pp.import_data_shift_rc_0157360(curve_num, shift_num)
    return rc_shift_records, rc_shift_dict


def train_loop(dataloader, model, optimizer, device, verbose=True):

    loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device, dtype=torch.float), y.to(device)

        # y_target = y[:, 0, model.adj_dis.shape[0]:].to(dtype=torch.float)
        # y_target_weights = y[:, 0, :model.adj_dis.shape[0]].to(dtype=torch.float)
        y_target = y[:, 0, model.adj_dis.shape[0]: (model.adj_dis.shape[0] + 1)].to(dtype=torch.float)
        y_target_weights = y[:, 0, 0: 1].to(dtype=torch.float)

        pred = model(x)
        pred = pred[:, :, 0]
        loss = loss_func_1(pred, y_target, y_target_weights)
        # loss = loss_func_2(pred, y_level)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            if verbose:
                print(f"loss: {loss: >7f}  [{current: >5d} / {size: >5d}]")


def val_loop(dataloader, train_dataloader, model, target_in_forward, device, add_loss_func=None, verbose=True):
    # loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.L1Loss()
    loss_func_3 = nn.MSELoss()
    loss_func_mape = add_loss_func

    model.eval()

    with torch.no_grad():
        x_list = []
        y_list = []
        for x, y in dataloader:
            x_list.append(x)
            y_list.append(y)
        x = torch.cat(x_list, dim=0).to(device, dtype=torch.float)
        y = torch.cat(y_list, dim=0).to(device)

        # y_target = y[:, 0, model.adj_dis.shape[0]:].to(dtype=torch.float)
        y_target = y[:, 0, model.adj_dis.shape[0]: (model.adj_dis.shape[0] + 1)].to(dtype=torch.float)
        # y_level_weights = y[:, :, 0].to(dtype=torch.float)
        pred = model(x)
        pred = pred[:, :, 0]

        if loss_func_mape:
            val_loss = loss_func_mape(y_target.cpu().numpy(), pred.cpu().numpy())
            val_metric = val_loss
            if verbose:
                print(f"Avg loss: {val_loss:>8f}")
                print(f"Avg MAPE: {val_metric.item():>8f} \n")
        else:
            # val_loss = loss_func_1(pred, y_level, y_level_weights).item()
            val_loss = loss_func_3(pred, y_target).item()
            val_metric = loss_func_2(pred[:, 0], y_target[:, 0])
            if verbose:
                print(f"Avg loss: {val_loss:>8f}")
                print(f"Avg MAE: {val_metric.item():>8f} \n")

    return val_loss, val_metric


def train_w_hp(
        trial, device, train_x, train_y, val_x, val_y,
        feat_in, k,
        adj_dis,
        target_in_forward, num_nodes, num_rep=1
):

    # batch_size = trial.suggest_int("batch_size", low=128, high=256, step=64)
    # lr = trial.suggest_float("lr", low=0.0003, high=0.0013, step=0.0002)
    # feat_out = trial.suggest_int("feat_out", low=16, high=80, step=32)
    # layer_out = trial.suggest_int("layer_out", low=2, high=4, step=2)
    batch_size = trial.suggest_int("batch_size", low=128, high=320, step=64)
    lr = trial.suggest_float("lr", low=0.0001, high=0.0005, step=0.0002)
    feat_out = trial.suggest_int("feat_out", low=16, high=112, step=32)
    layer_out = trial.suggest_int("layer_out", low=2, high=4, step=1)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = HomoDCRNN(feat_in, k, feat_out, layer_out, num_nodes)

        model.adj_dis = adj_dis
        model.name = 'LevelPredHomoDCRNN'
        optim = torch.optim.Adam([
            {'params': model.sdcrnn.parameters(), 'lr': lr},
            {'params': model.dense_readout.parameters(), 'lr': lr},
            {'params': model.gru.parameters(), 'lr': lr},
        ])

        val_metric, _ = mo.train(
            device,
            train_x, train_y, val_x, val_y,
            target_in_forward,
            model, train_loop, val_loop, batch_size, lr,
            optim=optim, trial=trial)

        val_metric_ave += val_metric
    objective_value = val_metric_ave / num_rep

    global best_score_optuna_tune
    label = 'HODCRNN'
    if objective_value < best_score_optuna_tune:
        best_score_optuna_tune = objective_value
        torch.save(
            {
                'model': model,
                'optimizer': optim.state_dict(),
            },
            f'{expr_dir_global}/best_{label}_optuna_tune_{objective_value}.pth'
        )
    return objective_value


def train_pred(
        df, df_precip, df_field, dict_rc, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, data_flood_stage, if_tune
):

    global expr_dir_global
    expr_dir_global = expr_dir

    # parameters - tune
    n_trials = 50
    tune_rep_num = 1

    # parameters - default model
    batch_size = 128
    lr = 0.0018
    feat_out = 64
    layer_out = 2

    feat_in = 1
    k = 3
    scaler = MinMaxScaler # StandardScaler

    # other parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reload model
    saved_model = torch.load(
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_1__2024-03-14-17-37-25/best_HODCRNN_optuna_tune_0.00023879233049228787.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_2__2024-03-14-17-34-33/best_HODCRNN_optuna_tune_0.0004181543772574514.pth'
        './outputs/experiments/ARCHIVE_pi_hodcrnn_3__2024-03-14-17-32-00/best_HODCRNN_optuna_tune_0.0005952782230451703.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_4__2024-03-14-17-29-51/best_HODCRNN_optuna_tune_0.0008744496735744178.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_5__2024-03-14-17-24-40/best_HODCRNN_optuna_tune_0.0010843131458386779.pth',
        # './outputs/experiments/ARCHIVE_pi_hodcrnn_6__2024-03-14-17-22-44/best_HODCRNN_optuna_tune_0.001381319249048829.pth',
    )
    model = saved_model['model']
    model.eval()
    model.to(device)
    model.name = 'LevelPredHomoDCRNN_tune'

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

    # precip, legacy, keep it
    # df_precip_scaled = ft.scale_precip_data(adj_matrix_dir, df_precip)
    # scaler_precip = scaler()
    # df_precip_normed = pd.DataFrame(
    #     scaler_precip.fit_transform(df_precip_scaled), columns=df_precip_scaled.columns, index=df_precip_scaled.index
    # )
    # df_precip_normed = df_precip_normed.rename(columns={0:'ave_precip'})

    # precip
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

    train_x = x[dataset_index[0]['train_index'], :, :][:, :, num_nodes:]
    train_y = y[dataset_index[0]['train_index'], :][:, :, :]
    val_x = x[dataset_index[0]['val_index'], :, :][:, :, num_nodes:]
    val_y = y[dataset_index[0]['val_index'], :][:, :, :]
    test_x = x[dataset_index[0]['test_index'], :, :][:, :, num_nodes:]
    test_y = y[dataset_index[0]['test_index'], :][:, :, :]
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    if num_test_sequences != test_y.shape[0]:
        raise ValueError('Test sets inconsistency.')

    # check field measurement dist among training and test sets
    num_action = len(test_df_field[test_df_field['water_level_adjusted'] > data_flood_stage['action'].values[0]])
    print(f"# of data points exceeding action stage in test set is {num_action}")
    if num_action == 0:
        with open('./outputs/USGS_gaga_filtering/gauge_delete_action_during_test.json', 'r') as file:
            gauge_delete_action_during_test = json.load(file)
        with open('./outputs/USGS_gaga_filtering/gauge_delete_action_during_test.json', 'w') as file:
            gauge_delete_action_during_test.append(target_gage)
            gauge_delete_action_during_test = list(set(gauge_delete_action_during_test))
            json.dump(gauge_delete_action_during_test, file)


    # train with hp tuning
    if if_tune and 'saved_model' not in locals():
        tag = 'pi_hodcrnn'
        study = optuna.create_study(direction="minimize",
                                    storage=f"sqlite:///tuner/db_level_pred_{tag}.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_level_pred_pi_hodcrnn.sqlite3 --port 8084
                                    study_name=f"level_pred_{tag}_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        study.optimize(
            lambda trial: train_w_hp(
                trial, device,
                train_x, train_y, val_x, val_y,
                feat_in, k,
                adj_dis,
                target_in_forward, num_nodes, tune_rep_num),
            n_trials=n_trials,
        )

        # get the best hps and close if_tune
        best_hps = study.best_trial.params
        batch_size = best_hps['batch_size']
        lr = best_hps['lr']
        feat_out = best_hps['feat_out']
        layer_out = best_hps['layer_out']

        # disable tune for training model using best hp
        if_tune = False

    # train without hp tuning
    if not if_tune:
        if 'saved_model' not in locals():
            model = HomoDCRNN(feat_in, k, feat_out, layer_out, num_nodes)

            model.name = 'LevelPredHomoDCRNN'
            model.adj_dis = adj_dis
            optim = torch.optim.Adam([
                {'params': model.sdcrnn.parameters(), 'lr': lr},
                {'params': model.dense_readout.parameters(), 'lr': lr},
                {'params': model.gru.parameters(), 'lr': lr},
            ])

            _, model = mo.train(device,
                                train_x, train_y, val_x, val_y,
                                target_in_forward,
                                model, train_loop, val_loop, batch_size, lr,
                                optim=optim)
            torch.save(model, f'{expr_dir}/model.pth')

        pred = mo.pred_4_test_hodcrnn(model, test_x, target_in_forward, device)
        pred = pred[:, 0, :]
        scaler_pred = scaler()
        scaler_pred.fit(df[[f"{target_gage}_00065"]])
        pred = scaler_pred.inverse_transform(pd.DataFrame(pred))[:, 0]
        pred = np.round(pred, 2)

        # modeled water level
        test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
        test_df_raw = df_raw[df_raw.index >= '2021-01-14'][[f'{target_gage}_00060', f'{target_gage}_00065']]
        test_df = test_df.rename(columns={
            f'{target_gage}_00060': 'modeled',
            f'{target_gage}_00065': 'water_level',
        })
        test_df_raw = test_df_raw.rename(columns={
            f'{target_gage}_00060': 'modeled',
            f'{target_gage}_00065': 'water_level',
        })

        # check the scope of reliable rating curve
        test_df_raw = mo.apply_rc_shifts(test_df_raw, convert_from='water_level')
        test_df_raw['diff'] = test_df_raw['modeled'] - test_df_raw['pred']
        test_df_raw = test_df_raw.resample('H', closed='right', label='right').mean()
        test_df_raw = test_df_raw[test_df_raw['diff'] == 0]

        # convert pred water level to discharge
        test_df['pred_water_level'] = pred
        test_df = mo.apply_rc_shifts(test_df, filter=test_df_raw, convert_from='pred_water_level')
        # test_df = test_df.reset_index()
        # test_df['pred'] = test_df.apply(
        #     lambda row: mo.approx_rc(
        #         row,
        #         df_raw[[f'{target_gage}_00065', f'{target_gage}_00060']],
        #         buffer_len=3,
        #         time_col_name='index', level_col_name='pred_water_level'
        #     ) if pd.isna(row['pred']) else row['pred'],
        #     axis=1
        # )
        # test_df = test_df.set_index('index')
        test_df_full = test_df.copy()

        # field
        test_df_field.index = test_df_field.index.ceil('H')
        test_df_field = test_df_field.groupby(level=0).mean()
        test_df['field'] = test_df_field['discharge']
        test_df = test_df[~test_df['field'].isna()]
        test_df = test_df.reset_index()
        test_df['pred'] = test_df.apply(
            lambda row: mo.approx_rc(
                row,
                df_raw[[f'{target_gage}_00065', f'{target_gage}_00060']],
                buffer_len=3,
                time_col_name='index', level_col_name='pred_water_level'
            ), axis=1
        )

    return test_df, test_df_full
