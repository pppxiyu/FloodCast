import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn, Tensor
import optuna

from sklearn.metrics import mean_absolute_percentage_error

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp


class MLP(nn.Module):
    def __init__(self, feat_in, feat_out, num_layer,):
        super().__init__()
        self.mlp = mo.stack_dense_layers(feat_in, feat_out, num_layer, activation=nn.Sigmoid())

    def forward(self, x_dis):
        return self.mlp(x_dis)

def train_loop(dataloader, model, optimizer, device):

    loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device, dtype=torch.float), y.to(device)

        y_target = y[:, 0, 5:].to(dtype=torch.float)  # hard coded
        y_target_weights = y[:, 0, :5].to(dtype=torch.float)  # hard coded

        pred = model(x)
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

        y_target = y[:, 0, 5:].to(dtype=torch.float)
        # y_level_weights = y[:, :, 0].to(dtype=torch.float)
        pred = model(x)

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
        trial, device, train_x, train_y, val_x, val_y,
        target_in_forward, num_rep=1
):

    batch_size = trial.suggest_int("batch_size", low=128, high=1024, step=256)
    lr = trial.suggest_float("lr", low=0.0005, high=0.0015, step=0.0002)
    num_layer = trial.suggest_int("num_layer", low=1, high=10, step=2)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = MLP(train_x.shape[1] * train_x.shape[2], train_x.shape[2], num_layer)
        model.name = 'mlp'
        optim = torch.optim.Adam([
            {'params': model.mlp.parameters(), 'lr': lr},
        ])
        val_metric, _ = mo.train(
            device,
            train_x.reshape(train_x.shape[0], -1), train_y,
            val_x.reshape(val_x.shape[0], -1), val_y,
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
    batch_size = 128
    lr = 0.0011
    num_layer = 1

    # other parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    df = df.resample('H', closed='right', label='right').mean()
    df_dis_normed = (df - df.min()) / (df.max() - df.min())
    dis_cols = [col for col in df.columns if col.endswith('00060')]
    df_dis_normed = df_dis_normed[dis_cols]
    for col in df_dis_normed:
        if col.endswith('00060'):
            df_dis_normed = pp.sample_weights(df_dis_normed, col, if_log=True)

    # inputs
    target_in_forward = 1
    inputs = (
            sorted([col for col in df_dis_normed if "_weights" in col], reverse=True)
            + sorted([col for col in dis_cols if "_weights" not in col], reverse=True)
    )

    # make sequences and remove samples with nan values
    df_dis_normed['index'] = range(len(df_dis_normed))
    sequences_w_index = ft.create_sequences(df_dis_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # index split for major data
    test_percent_updated, test_df_field, num_test_sequences = ft.update_test_percent(df_field, df_dis_normed,
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
                                    storage=f"sqlite:///tuner/db_dis_pred_mlp.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_mlp.sqlite3 --port 8080
                                    study_name=f"dis_pred_mlp_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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
        num_layer = best_hps['num_layer']

        # disable tune for training model using best hp
        if_tune = False

    # train without hp tuning
    if not if_tune:

        model = MLP(train_x.shape[1] * train_x.shape[2], train_x.shape[2], num_layer)
        model.name = 'mlp'
        optim = torch.optim.Adam([
            {'params': model.mlp.parameters(), 'lr': lr},
        ])
        _, model = mo.train(device,
                            train_x.reshape(train_x.shape[0], -1), train_y,
                            val_x.reshape(val_x.shape[0], -1), val_y,
                            target_in_forward,
                            model, train_loop, val_loop, batch_size, lr,
                            optim=optim)

        pred = mo.pred_4_test_hodcrnn(model, test_x.reshape(test_x.shape[0], -1), target_in_forward, device)
        pred = pred[:, 0]
        pred = (
                pred * (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min())
                + df[f"{target_gage}_00060"].min()
        )

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
