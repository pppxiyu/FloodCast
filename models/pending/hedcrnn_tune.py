import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import optuna

from sklearn.metrics import mean_absolute_percentage_error

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

import functools


expr_dir_global = ''
best_score_optuna_tune = 1e10


def train_loop(dataloader, model, optimizer, device):

    loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device, dtype=torch.float), y.to(device)

        x_dis = x[:, :, :5].unsqueeze(-1)  # hard coded here
        x_precip = x[:, :, 5:].unsqueeze(-1)  # hard coded here

        # it_rains = (x_precip != 0).float()
        # x_precip = torch.cat((x_precip, it_rains), dim=-1)

        if y.shape[2] <= 2:
            y_target = y[:, :, 1:2].to(dtype=torch.float)  # hard coded here
            y_target_weights = y[:, :, 0:1].to(dtype=torch.float)  # hard coded here

            pred = model(x_dis, x_precip)
            pred = pred[:, 0:1, :]  # hard coded here
            loss = loss_func_1(pred, y_target, y_target_weights)
            # loss = loss_func_2(pred, y_level)
        else:
            y_target = y[:, 0, 5:, np.newaxis].to(dtype=torch.float)  # hard coded here
            y_target_weights = y[:, 0, :5, np.newaxis].to(dtype=torch.float)  # hard coded here

            pred = model(x_dis, x_precip)
            # pred = pred[:, :5, 0]  # hard coded here
            loss = loss_func_1(pred, y_target, y_target_weights)
            # loss = loss_func_2(pred, y_level)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss: >7f}  [{current: >5d} / {size: >5d}]")


def val_loop(dataloader, train_dataloader, model, target_in_forward, device, add_loss_func=None):
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

        x_dis = x[:, :, :5].unsqueeze(-1)  # hard coded here
        x_precip = x[:, :, 5:].unsqueeze(-1)  # hard coded here

        # it_rains = (x_precip != 0).float()
        # x_precip = torch.cat((x_precip, it_rains), dim=-1)

        if y.shape[2] <= 2:
            y_target = y[:, :, 1].to(dtype=torch.float)  # hard coded here

            pred = model(x_dis, x_precip)
            pred = pred[:, 0:1, 0]  # hard coded here
        else:
            y_target = y[:, 0, 5:, np.newaxis].to(dtype=torch.float)  # hard coded here
            pred = model(x_dis, x_precip)
            # pred = pred[:, :5, 0]

        if loss_func_mape:
            val_loss = loss_func_mape(y_target.cpu().numpy(), pred.cpu().numpy())
            val_metric = val_loss
            print(f"Avg loss: {val_loss:>8f}")
            print(f"Avg MAPE: {val_metric.item():>8f} \n")
        else:
            # val_loss = loss_func_1(pred, y_level, y_level_weights).item()
            val_loss = loss_func_3(pred, y_target).item()
            val_metric = loss_func_2(pred[:, 0], y_target[:, 0])
            print(f"Avg loss: {val_loss:>8f}")
            print(f"Avg MAE: {val_metric.item():>8f} \n")

    return val_loss, val_metric


def train_w_hp(
        trial, model, device,
        train_x_tune, train_y_tune, val_x_tune, val_y_tune,
        feat_in,
        fine_tune_rep_num, tune_mode,
        target_in_forward, num_rep=1
):

    batch_size = trial.suggest_int("batch_size", low=8, high=24, step=8)
    lr = trial.suggest_float("lr", low=0.0001, high=0.00015, step=0.0002)
    num_layers = trial.suggest_int("num_layers", low=1, high=4, step=1)

    best_val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        best_val_metric = 1e8
        for _ in range(fine_tune_rep_num):
            if tune_mode == 'w_out':
                model.dense_flex_1 = mo.stack_dense_layers(
                    1, 1, num_layers, activation=nn.Sigmoid()
                )
                for ly in model.dense_flex_1:
                    if isinstance(ly, nn.Linear):
                        nn.init.normal_(ly.weight, mean=0.0, std=0.5)
                        nn.init.normal_(ly.bias, mean=0.0, std=5)

            elif tune_mode == 'w_h':
                model.dense_flex_1 = mo.stack_dense_layers(feat_in, 1, num_layers)
                for ly in model.dense_flex_1:
                    if isinstance(ly, nn.Linear):
                        nn.init.normal_(ly.weight, mean=0.0, std=0.001)
                        nn.init.normal_(ly.bias, mean=0.0, std=0.01)

            val_metric, _ = mo.train(
                device,
                train_x_tune, train_y_tune, val_x_tune, val_y_tune,
                target_in_forward,
                model, train_loop,
                functools.partial(val_loop,
                                  add_loss_func=mean_absolute_percentage_error),
                batch_size, None,
                optim=torch.optim.Adam([{
                    'params': model.dense_flex_1.parameters(), 'lr': lr
                }]))
            best_val_metric = min(best_val_metric, val_metric)

        best_val_metric_ave += best_val_metric
    return best_val_metric_ave / num_rep


def train_pred(
        df, df_precip, df_field, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    saved = torch.load('best_HEDCRNN_optuna_tune_0.0003040653245989233.pth')
    model = saved['model']
    optim = saved['optimizer']
    model.to(device)

    # parameters - tune
    n_trials = 50
    tune_mode = 'w_h'
    fine_tune_rep_num = 5

    # parameters - default model
    if tune_mode == 'w_h':
        batch_size = 8
        lr = 0.001
        feat_in = model.dense_readout[0].weight.shape[1]
        num_layers = 3
    elif tune_mode == 'w_out':
        batch_size = 8
        lr = 0.001
        num_layers = 3

    # data
    adj_precip = pd.read_csv(f'{adj_matrix_dir}/adj_matrix_precipitation.csv', index_col=0)

    df = df.resample('H', closed='right', label='right').mean()
    df_dis_normed = (df - df.min()) / (df.max() - df.min())
    dis_cols = [col for col in df.columns if col.endswith('00060')]
    df_dis_normed = df_dis_normed[dis_cols]
    for col in df_dis_normed:
        if col.endswith('00060'):
            df_dis_normed = pp.sample_weights(df_dis_normed, col, if_log=True)

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
    df_precip_normed = ((df_precip_scaled - df_precip_scaled.min().min())
                        / (df_precip_scaled.max().max() - df_precip_scaled.min().min()))
    df_precip_normed = df_precip_normed[
        [f"clat{str(round(float(i.split('_')[0]) - 0.05, 1))}_clon{str(round(float(i.split('_')[1]) - 0.05, 1))}"
         for i in adj_precip.columns]
    ]

    df_normed = pd.concat([df_dis_normed, df_precip_normed], axis=1)

    # inputs
    target_in_forward = 1
    inputs = (
            sorted([col for col in df_dis_normed if "_weights" in col], reverse=True)
            + sorted([col for col in dis_cols if "_weights" not in col], reverse=True)
            + list(df_precip_normed.columns)
    )

    # make sequences and remove samples with nan values
    df_normed['index'] = range(len(df_normed))
    sequences_w_index = ft.create_sequences(df_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # index split for major data
    test_percent_updated, test_df_field, num_test_sequences = ft.update_test_percent(df_field, df_normed,
                                                                                     sequences_w_index, test_percent)
    x = sequences_w_index[:, :, :-1][:, :-1, :]
    dataset_index = ft.create_index_4_cv(x, False, None,
                                         val_percent, test_percent_updated, None, None)  # codes for cv is not revised

    # make datasets
    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]  # hard coded here
    y = sequences_w_index[:, :, :-1][:, -len(forward):, :]  # hard coded here
    y_index = sequences_w_index[:, :, [-1]][:, -len(forward):, :]  # hard coded here

    test_x = x[dataset_index[0]['test_index'], :, :][:, :, 5:]  # hard coded here
    test_y = y[dataset_index[0]['test_index'], :][:, :, :10]  # hard coded here
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    # data for tuning
    train_df_field, val_df_field = ft.split_df_field(df_field, test_df_field, val_percent, test_percent_updated)
    train_df_field, val_df_field = ft.filter_df_field(train_df_field, val_df_field,
                                                      df_normed, sequences_w_index[:, -1, -1])
    train_index_seq_field, val_index_seq_field = ft.get_sequence_indices(train_df_field, val_df_field,
                                                                         df_normed, sequences_w_index[:, -1, -1])

    train_df_field.insert(0, 'discharge_weights', train_df_field.pop('discharge_weights'))
    # train_df_field['discharge'] = ((train_df_field['discharge'] - df[f"{target_gage}_00060"].min()) /
    #                                (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min()))
    val_df_field.insert(0, 'discharge_weights', val_df_field.pop('discharge_weights'))
    # val_df_field['discharge'] = ((val_df_field['discharge'] - df[f"{target_gage}_00060"].min()) /
    #                                (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min()))
    train_x_tune = x[train_index_seq_field, :, :][:, :, 5:]  # hard coded here
    train_y_tune = train_df_field.values[:, np.newaxis, :]  # hard coded here
    val_x_tune = x[val_index_seq_field, :, :][:, :, 5:]  # hard coded here
    val_y_tune = val_df_field.values[:, np.newaxis, :]  # hard coded here

    assert num_test_sequences == test_y.shape[0], 'Test sets inconsistency.'

    # model attr
    model.norm_param = [df[f"{target_gage}_00060"].min(), df[f"{target_gage}_00060"].max()]
    model.tune_mode = tune_mode
    model.name = 'DisPredHeteroDCRNN_tune'

    # train with hp tuning
    if if_tune:


        study = optuna.create_study(direction="minimize",
                                    storage="sqlite:///tuner/db_dis_pred_hedcrnn_tune.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hedcrnn_tune.sqlite3 --port 8088
                                    study_name="dis_pred_hedcrnn_tune_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        study.optimize(
            lambda trial: train_w_hp(
                trial, model, device,
                train_x_tune, train_y_tune, val_x_tune, val_y_tune,
                feat_in,
                fine_tune_rep_num, tune_mode,
                target_in_forward, num_rep=1
            ),
            n_trials=n_trials,
        )

        # get the best hps and close if_tune
        best_hps = study.best_trial.params
        batch_size = best_hps['batch_size']
        lr = best_hps['lr']
        num_layers = best_hps['num_layers']

        # disable tune for training model using best hp
        if_tune = False

    # train without hp tuning
    if not if_tune:

        # # fine tune
        # best_val_metric_mape = 1e8
        # best_model = None
        # for _ in range(fine_tune_rep_num):
        #     if tune_mode == 'w_out':
        #         model.dense_flex_1 = mo.stack_dense_layers(
        #             1, 1, num_layers,
        #         )
        #         for ly in model.dense_flex_1:
        #             if isinstance(ly, nn.Linear):
        #                 nn.init.normal_(ly.weight, mean=0.0, std=0.5)
        #                 nn.init.normal_(ly.bias, mean=0.0, std=5)
        #
        #     elif tune_mode == 'w_h':
        #         model.dense_flex_1 = mo.stack_dense_layers(feat_in, 1, num_layers)
        #         for ly in model.dense_flex_1:
        #             if isinstance(ly, nn.Linear):
        #                 nn.init.normal_(ly.weight, mean=0.0, std=0.001)
        #                 nn.init.normal_(ly.bias, mean=0.0, std=0.01)
        #
        #     val_metric_mape, model = mo.train(
        #         device,
        #         train_x_tune, train_y_tune, val_x_tune, val_y_tune,
        #         target_in_forward,
        #         model, train_loop,
        #         functools.partial(val_loop,
        #                           add_loss_func=mean_absolute_percentage_error),
        #         batch_size, None,
        #         optim=torch.optim.Adam([{
        #             'params': model.dense_flex_1.parameters(), 'lr': lr
        #         }]))
        #
        #     if val_metric_mape < best_val_metric_mape:
        #         best_val_metric_mape = val_metric_mape
        #         best_model = model
        #
        # pred_y_tune = mo.pred_4_test_hedcrnn(best_model, test_x, target_in_forward, device)
        # pred_y_tune = pred_y_tune[:, 0, :]
        pred_y_tune = mo.pred_4_test_hedcrnn(model, test_x, target_in_forward, device)
        pred_y_tune = pred_y_tune * (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min()) + df[f"{target_gage}_00060"].min()
        pred_y_tune = pred_y_tune[:, 0, :]

        # modeled discharge
        test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
        test_df = test_df.rename(columns={
            f'{target_gage}_00060': 'modeled',
            f'{target_gage}_00065': 'water_level',
        })

        # pred discharge
        test_df['pred'] = pred_y_tune
        test_df_full = test_df.copy()

        # field discharge
        test_df_field.index = test_df_field.index.ceil('H')
        test_df_field = test_df_field.groupby(level=0).mean()
        test_df['field'] = test_df_field['discharge']
        test_df = test_df[~test_df['field'].isna()]

    return test_df, test_df_full
