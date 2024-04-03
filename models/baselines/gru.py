import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import optuna

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

import warnings

class GRU(nn.Module):
    def __init__(
            self,
            feat_in,
            feat_out,
            num_readout_layer,
            num_out_site,
            num_forward=1
    ):
        super().__init__()

        self.gru = torch.nn.GRU(feat_in, feat_out, batch_first=True)
        self.num_forward = num_forward
        self.dense_readout = mo.stack_dense_layers(feat_out, num_forward * num_out_site, num_readout_layer)

    def forward(self, x):
        o, h = self.gru(x)
        h = h[0, :, :, ]
        out = self.dense_readout(h)
        return out.unsqueeze(1)


def train_loop(dataloader, model, optimizer, device):

    loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device, dtype=torch.float), y.to(device)

        # y_level = y[:, :, 5:10].to(dtype=torch.float)
        # y_level_weights = y[:, :, :5].to(dtype=torch.float)
        y_level = y[:, :, 5: 6].to(dtype=torch.float)
        y_level_weights = y[:, :, 0: 1].to(dtype=torch.float)

        pred = model(x)
        loss = loss_func_1(pred, y_level, y_level_weights)
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

        # y_level = y[:, :, 5:10].to(dtype=torch.float)
        y_level = y[:, :, 5:6].to(dtype=torch.float)
        # y_level_weights = y[:, :, 0].to(dtype=torch.float)
        pred = model(x)

        if loss_func_mape:
            val_loss = loss_func_mape(y_level.cpu().numpy(), pred.cpu().numpy())
            val_metric = val_loss
            print(f"Avg loss: {val_loss:>8f}")
            print(f"Avg MAPE: {val_metric.item():>8f} \n")
        else:
            # val_loss = loss_func_1(pred, y_level, y_level_weights).item()
            val_loss = loss_func_3(pred, y_level).item()
            val_metric = loss_func_2(pred, y_level)
            print(f"Avg loss: {val_loss:>8f}")
            print(f"Avg MAE: {val_metric.item():>8f} \n")

    return val_loss, val_metric


def train_w_hp(
        trial, device, train_x, train_y, val_x, val_y,
        target_in_forward, num_rep=1
):

    batch_size = trial.suggest_int("batch_size", low=128, high=640, step=128)
    lr = trial.suggest_float("lr", low=0.0002, high=0.0018, step=0.0004)
    feat_out = trial.suggest_int("feat_out", low=16, high=64, step=16)
    layer_out = trial.suggest_int("layer_out", low=2, high=3, step=1)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = GRU(train_x.shape[2], feat_out, layer_out, 1)
        model.name = 'DisPredGRU'
        optim = torch.optim.Adam([
            {'params': model.gru.parameters(), 'lr': lr},
            {'params': model.dense_readout.parameters(), 'lr': lr},
        ])

        val_metric, _ = mo.train(
            device,
            train_x, train_y, val_x, val_y,
            target_in_forward,
            model, train_loop, val_loop, batch_size, lr,
            optim=optim, trial=trial)

        val_metric_ave += val_metric
    return val_metric_ave / num_rep


def train_pred(
        df, df_precip, df_field, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, if_tune
):

    # parameters - tune
    n_trials = 50
    tune_rep_num = 1

    # parameters - default model
    batch_size = 640
    lr = 0.001
    feat_out = 16
    layer_out = 2

    # other parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = PowerTransformer

    # data
    df = df.resample('H', closed='right', label='right').mean()
    scaler_stream = scaler()
    df_dis_normed = pd.DataFrame(scaler_stream.fit_transform(df), columns=df.columns, index=df.index)
    dis_cols = [col for col in df.columns if col.endswith('00060')]
    df_dis_normed = df_dis_normed[dis_cols]
    for col in df_dis_normed:
        if col.endswith('00060'):
            df_dis_normed = pp.sample_weights(df_dis_normed, col, if_log=True)

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
        df_dis_normed,
        df_precip_normed
    ], axis=1)

    # inputs
    target_in_forward = 1
    inputs = (
            sorted([col for col in df_normed if "_weights" in col], reverse=True)
            + sorted([col for col in dis_cols if "_weights" not in col], reverse=True)
            + [df_precip_normed.columns[0]]
    )

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
    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]  # hard coded here
    y = sequences_w_index[:, :, :-1][:, -len(forward):, :]  # hard coded here
    y_index = sequences_w_index[:, :, [-1]][:, -len(forward):, :]  # hard coded here

    train_x = x[dataset_index[0]['train_index'], :, :][:, :, 5:]  # hard coded here
    train_y = y[dataset_index[0]['train_index'], :][:, :, :]  # hard coded here
    val_x = x[dataset_index[0]['val_index'], :, :][:, :, 5:]  # hard coded here
    val_y = y[dataset_index[0]['val_index'], :][:, :, :]  # hard coded here
    test_x = x[dataset_index[0]['test_index'], :, :][:, :, 5:]  # hard coded here
    test_y = y[dataset_index[0]['test_index'], :][:, :, :]  # hard coded here
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    if num_test_sequences != test_y.shape[0]:
        raise ValueError('Test sets inconsistency.')

    # train with hp tuning
    if if_tune:
        study = optuna.create_study(direction="minimize",
                                    storage=f"sqlite:///tuner/db_dis_pred_gru.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_gru.sqlite3 --port 8081
                                    study_name=f"dis_pred_gru_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        study.optimize(
            lambda trial: train_w_hp(
                trial, device,
                train_x, train_y, val_x, val_y,
                target_in_forward, tune_rep_num),
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
        model = GRU(train_x.shape[2], feat_out, layer_out, 1)
        model.name = 'DisPredGRU'

        optim = torch.optim.Adam([
            {'params': model.gru.parameters(), 'lr': lr},
            {'params': model.dense_readout.parameters(), 'lr': lr},
        ])

        _, model = mo.train(
            device,
            train_x, train_y, val_x, val_y,
            target_in_forward,
            model, train_loop, val_loop, batch_size, lr,
            optim=optim)

        pred = mo.pred_4_test_hodcrnn(model, test_x, target_in_forward, device)
        pred = pred[:, :, 0]
        scaler_pred = scaler()
        scaler_pred.fit(df[[f"{target_gage}_00060"]])
        pred = scaler_pred.inverse_transform(pd.DataFrame(pred))[:, 0]

        # modeled discharge
        test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
        test_df = test_df.rename(columns={
            f'{target_gage}_00060': 'modeled',
            f'{target_gage}_00065': 'water_level',
        })

        # pred discharge
        test_df['pred'] = pred
        test_df_full = test_df.copy()

        # field discharge
        test_df_field.index = test_df_field.index.ceil('H')
        test_df_field = test_df_field.groupby(level=0).mean()
        test_df['field'] = test_df_field['discharge']
        test_df = test_df[~test_df['field'].isna()]

    return test_df, test_df_full
