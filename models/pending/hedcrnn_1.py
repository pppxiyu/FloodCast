import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.init as init
import optuna
from models.hodcrnn import DConv

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp


class MultiDCRNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K1: int, K2: int, p: float, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K1 = K1
        self.K2 = K2
        self.bias = bias

        self._create_parameters_and_layers()

        self.proj = mo.stack_dense_layers(out_channels, out_channels, 1)

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z_s = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K1,
            bias=self.bias,
        )
        self.conv_x_z_p = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )
        self.conv_x_z_p2s = DConv(
            # in_channels=self.in_channels + self.out_channels,
            # in_channels=self.in_channels,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r_s = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K1,
            bias=self.bias,
        )
        self.conv_x_r_p = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )
        self.conv_x_r_p2s = DConv(
            # in_channels=self.in_channels + self.out_channels,
            # in_channels=self.in_channels,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h_s = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K1,
            bias=self.bias,
        )
        self.conv_x_h_p = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )
        self.conv_x_h_p2s = DConv(
            # in_channels=self.in_channels + self.out_channels,
            # in_channels=self.in_channels,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K2,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate_enhanced(
            self, X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H
    ):
        Z_s = torch.cat([X_s, H[:, : X_s.shape[1], :]], dim=-1)
        # Z_s = torch.cat([X_s, H], dim=-1)
        Z_s = self.conv_x_z_s(Z_s, edge_index_s, edge_weight_s)

        Z_p = torch.cat([X_p, H[:, -X_p.shape[1]:, :]], dim=-1)
        Z_p = self.conv_x_z_p(Z_p, edge_index_p, edge_weight_p)

        # Z_p2s = torch.cat([X_p, H[:, -X_p.shape[1]:, :]], dim=-1)
        # Z_p2s = torch.cat([X_p, H], dim=-1)
        # Z_p2s = self.conv_x_z_p2s(Z_p2s, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # Z_p2s = self.conv_x_z_p2s(X_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # Z_p2s = self.conv_x_z_p2s(Z_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)

        Z = torch.sigmoid(
            torch.cat((
                Z_s
                # + Z_p2s
                # + self.proj(Z_p2s)
                , Z_p
            ), dim=-2))
        # Z = torch.sigmoid(
        #     Z_s
        #     + Z_p2s
        # )
        return Z

    def _calculate_reset_gate_enhanced(
            self, X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H
    ):
        R_s = torch.cat([X_s, H[:, : X_s.shape[1], :]], dim=-1)
        # R_s = torch.cat([X_s, H], dim=-1)
        R_s = self.conv_x_r_s(R_s, edge_index_s, edge_weight_s)

        R_p = torch.cat([X_p, H[:, -X_p.shape[1]:, :]], dim=-1)
        R_p = self.conv_x_r_p(R_p, edge_index_p, edge_weight_p)

        # R_p2s = torch.cat([X_p, H[:, -X_p.shape[1]:, :]], dim=-1)
        # R_p2s = torch.cat([X_p, H], dim=-1)
        # R_p2s = self.conv_x_r_p2s(R_p2s, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # R_p2s = self.conv_x_r_p2s(X_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # R_p2s = self.conv_x_r_p2s(R_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)

        R = torch.sigmoid(
            torch.cat((
                R_s
                # + R_p2s
                # + self.proj(R_p2s)
                , R_p
            ), dim=-2))
        # R = torch.sigmoid(
        #     R_s
        #     + R_p2s
        # )
        return R

    def _calculate_candidate_state_enhanced(
            self, X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H, R,
    ):
        gated_H = H * R
        H_tilde_s = torch.cat([X_s, gated_H[:, : X_s.shape[1], :]], dim=-1)
        # H_tilde_s = torch.cat([X_s, gated_H], dim=-1)
        H_tilde_s = self.conv_x_h_s(H_tilde_s, edge_index_s, edge_weight_s)

        H_tilde_p = torch.cat([X_p, gated_H[:, -X_p.shape[1]:, :]], dim=-1)
        H_tilde_p = self.conv_x_h_p(H_tilde_p, edge_index_p, edge_weight_p)

        # H_tilde_p2s = torch.cat([X_p, gated_H[:, -X_p.shape[1]:, :]], dim=-1)
        # H_tilde_p2s = torch.cat([X_p, gated_H], dim=-1)
        # H_tilde_p2s = self.conv_x_h_p2s(H_tilde_p2s, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # H_tilde_p2s = self.conv_x_h_p2s(X_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)
        # H_tilde_p2s = self.conv_x_h_p2s(H_tilde_p, edge_index_p, edge_weight_p, edge_index_p2s, edge_weight_p2s)

        H_tilde = torch.tanh(
            torch.cat((
                H_tilde_s
                # + H_tilde_p2s
                # + self.proj(H_tilde_p2s)
                , H_tilde_p
            ), dim=-2))
        # H_tilde = torch.tanh(
        #     H_tilde_s
        #     # + H_tilde_p2s
        # )
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + ((1 - Z) * H_tilde)
        return H

    def forward(
        self,
        X_s: torch.FloatTensor,
        X_p: torch.FloatTensor,
        edge_index_s: torch.LongTensor,
        edge_weight_s: torch.FloatTensor,
        edge_index_p: torch.LongTensor,
        edge_weight_p: torch.FloatTensor,
        edge_index_p2s: torch.LongTensor,
        edge_weight_p2s: torch.FloatTensor,
        H: torch.FloatTensor = None,
    ):

        H = self._set_hidden_state(torch.cat((
            X_s[:, :, 0],
            X_p[:, :, 0]
        ), dim=-1), H)
        # H = self._set_hidden_state(X_s[:, :, 0], H)
        Z = self._calculate_update_gate_enhanced(
            X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H,
        )
        R = self._calculate_reset_gate_enhanced(
            X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H,
        )
        H_tilde = self._calculate_candidate_state_enhanced(
            X_s, X_p,
            edge_index_s, edge_weight_s,
            edge_index_p, edge_weight_p,
            edge_index_p2s, edge_weight_p2s,
            H, R,
        )
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class HeteroDCRNN(nn.Module):
    def __init__(
            self,
            feat_in_dis, feat_in_precip, feat_out_proj,
            K_dis, K_precip, feat_out_dcrnn,
            num_readout_layer,
            dropout=1,
            num_layer_proj_dis=1, num_layer_proj_precip=1, num_forward=1
    ):
        super().__init__()

        self.proj_dis = mo.stack_dense_layers(feat_in_dis, feat_out_proj, num_layer_proj_dis, bias=False)
        self.proj_precip = mo.stack_dense_layers(feat_in_precip, feat_out_proj, num_layer_proj_precip, bias=False)
        for layer in self.proj_dis:
            init.uniform_(layer.weight, a=.75, b=1.)
        for layer in self.proj_precip:
            init.uniform_(layer.weight, a=1., b=2.)

        self.mdcrnn = MultiDCRNN(feat_out_proj, feat_out_dcrnn, K_dis, K_precip, dropout, bias=True)

        self.num_forward = num_forward
        self.dense_readout = mo.stack_dense_layers(feat_out_dcrnn, num_forward * 1, num_readout_layer)

        self.dense_flex_1 = None
        self.dense_flex_2 = None

        self.aggr_nodes = mo.stack_dense_layers(28, 5, 1)

    def forward(self, x_dis, x_precip):

        assert x_dis.shape[1] == x_precip.shape[1]

        # data prep
        edge_index_dis = np.vstack(np.where(self.adj_dis.to_numpy() != 0))
        edge_index_dis = torch.from_numpy(edge_index_dis).to(torch.long).to(x_dis.device)
        edge_weight_dis = self.adj_dis.to_numpy()[np.where(self.adj_dis != 0)[0], np.where(self.adj_dis != 0)[1]]
        edge_weight_dis = torch.from_numpy(edge_weight_dis).float().to(x_dis.device)

        edge_index_precip = np.vstack(np.where(self.adj_precip.to_numpy() != 0))
        edge_index_precip = torch.from_numpy(edge_index_precip).to(torch.long).to(x_dis.device)
        edge_weight_precip = self.adj_precip.to_numpy()[
            np.where(self.adj_precip != 0)[0], np.where(self.adj_precip != 0)[1]
        ]
        edge_weight_precip = torch.from_numpy(edge_weight_precip).float().to(x_dis.device)

        edge_index_precip2dis = np.vstack(np.where(self.adj_precip2dis.to_numpy() != 0))
        edge_index_precip2dis = torch.from_numpy(edge_index_precip2dis).to(torch.long).to(x_dis.device)
        edge_weight_precip2dis = self.adj_precip2dis.to_numpy()[
            np.where(self.adj_precip2dis != 0)[0], np.where(self.adj_precip2dis != 0)[1]
        ]
        edge_weight_precip2dis = torch.from_numpy(edge_weight_precip2dis).float().to(x_dis.device)

        # projection
        # x_dis_proj = self.proj_dis(x)
        # x_precip_proj = self.proj_precip(x_precip)
        x_dis_proj = x_dis
        x_precip_proj = x_precip

        # multi-diffusion convolution rnn
        h = None
        for i in range(x_dis.shape[1]):
            h = self.mdcrnn(
                x_dis_proj[:, i, :, :], x_precip_proj[:, i, :, :],
                edge_index_dis, edge_weight_dis,
                edge_index_precip, edge_weight_precip,
                edge_index_precip2dis, edge_weight_precip2dis,
                h
            )

        # readout
        out = self.dense_readout(h)
        out = self.aggr_nodes(out[:,:,0]).unsqueeze(-1)
        # if self.dense_flex_1 is not None:
        #     tu_1 = self.dense_flex_1(h)
        #     out += tu_1
        #
        # if self.dense_flex_2 is not None:
        #     tu_2 = self.dense_flex_2(h)
        #     out += tu_2

        return out


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

        y_target = y[:, 0, 5:].to(dtype=torch.float)  # hard coded here
        y_target_weights = y[:, 0, :5].to(dtype=torch.float)  # hard coded here

        pred = model(x_dis, x_precip)
        pred = pred[:, :5, 0]  # hard coded here
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

        y_target = y[:, 0, 5:].to(dtype=torch.float)  # hard coded here
        pred = model(x_dis, x_precip)
        pred = pred[:, :5, 0]

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
        feat_in_dis, feat_in_precip, k_dis, k_precip,
        adj_dis, adj_precip, adj_precip2dis,
        dropout,
        target_in_forward, num_rep=1
):

    batch_size = trial.suggest_int("batch_size", low=128, high=192, step=64)
    lr = trial.suggest_float("lr", low=0.0016, high=0.0022, step=0.0002)
    # feat_proj = trial.suggest_int("feat_proj", low=1, high=3, step=1)
    feat_out = trial.suggest_int("feat_out", low=32, high=64, step=16)
    layer_out = trial.suggest_int("layer_out", low=2, high=2.1, step=1)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = HeteroDCRNN(
            feat_in_dis, feat_in_precip,  1,
            k_dis, k_precip, feat_out,
            layer_out, dropout=dropout)

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

            model.name = 'DisPredHeteroDCRNN2'
            model.module.adj_dis = adj_dis
            model.module.adj_precip = adj_precip
            model.module.adj_precip2dis = adj_precip2dis

            optim = torch.optim.Adam([
                {'params': model.module.mdcrnn.parameters(), 'lr': lr},
                {'params': model.module.dense_readout.parameters(), 'lr': lr},
                {'params': model.module.proj_dis.parameters(), 'lr': lr},
                {'params': model.module.proj_precip.parameters(), 'lr': lr},
                {'params': model.module.aggr_nodes.parameters(), 'lr': lr},
            ])

        else:
            model.adj_dis = adj_dis
            model.adj_precip = adj_precip
            model.adj_precip2dis = adj_precip2dis
            model.name = 'DisPredHeteroDCRNN2'

            optim = torch.optim.Adam([
                {'params': model.mdcrnn.parameters(), 'lr': lr},
                {'params': model.dense_readout.parameters(), 'lr': lr},
                {'params': model.proj_dis.parameters(), 'lr': lr},
                {'params': model.proj_precip.parameters(), 'lr': lr},
                {'params': model.aggr_nodes.parameters(), 'lr': lr},
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
    batch_size = 512
    lr = 0.002
    feat_in_dis = 1
    feat_in_precip = 1
    k_dis, k_precip = 2, 3
    feat_proj = 1
    feat_out = 16
    layer_out = 2
    dropout = 0

    # other parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    adj_dis = pd.read_csv(f'{adj_matrix_dir}/adj_matrix.csv', index_col=0)
    adj_precip = pd.read_csv(f'{adj_matrix_dir}/adj_matrix_precipitation.csv', index_col=0)
    adj_precip2dis = pd.read_csv(f'{adj_matrix_dir}/adj_matrix_both.csv', index_col=0)

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
    df_precip_normed = (df_precip_scaled - df_precip_scaled.min()) / (df_precip_scaled.max() - df_precip_scaled.min())
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

    train_x = x[dataset_index[0]['train_index'], :, :][:, :, 5:]  # hard coded here
    train_y = y[dataset_index[0]['train_index'], :][:, :, :10]  # hard coded here
    val_x = x[dataset_index[0]['val_index'], :, :][:, :, 5:]  # hard coded here
    val_y = y[dataset_index[0]['val_index'], :][:, :, :10]  # hard coded here
    test_x = x[dataset_index[0]['test_index'], :, :][:, :, 5:]  # hard coded here
    test_y = y[dataset_index[0]['test_index'], :][:, :, :10]  # hard coded here
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    # # shuffle for mixing precip
    # indices = np.arange(len(train_x))
    # np.random.shuffle(indices)
    # train_x = train_x[indices]
    # train_y = train_y[indices]

    # delete after developing
    train_x = train_x[-train_x.shape[0] // 5:, :, :]
    train_y = train_y[-train_y.shape[0] // 5:, :, :]
    val_x = val_x[-val_x.shape[0] // 5:, :, :]
    val_y = val_y[-val_y.shape[0] // 5:, :, :]

    assert num_test_sequences == test_y.shape[0] ,'Test sets inconsistency.'

    # train with hp tuning
    if if_tune:
        study = optuna.create_study(direction="minimize",
                                    storage="sqlite:///tuner/db_dis_pred_hedcrnn1.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hedcrnn.sqlite3 --port 8084
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hedcrnn1.sqlite3 --port 8085
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hedcrnn2.sqlite3 --port 8086
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hedcrnn3.sqlite3 --port 8087
                                    study_name="dis_pred_hedcrnn1_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        study.optimize(
            lambda trial: train_w_hp(
                trial, device,
                train_x, train_y, val_x, val_y,
                feat_in_dis, feat_in_precip, k_dis, k_precip,
                adj_dis, adj_precip, adj_precip2dis, dropout,
                target_in_forward, tune_rep_num),
            n_trials=n_trials,
        )

        # get the best hps and close if_tune
        best_hps = study.best_trial.params
        batch_size = best_hps['batch_size']
        lr = best_hps['lr']
        feat_proj = best_hps['feat_proj']
        feat_out = best_hps['feat_out']
        layer_out = best_hps['layer_out']

        # disable tune for training model using best hp
        if_tune = False

    # train without hp tuning
    if not if_tune:

        model = HeteroDCRNN(
            feat_in_dis, feat_in_precip, feat_proj,
            k_dis, k_precip, feat_out,
            layer_out, dropout=dropout)

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

            model.name = 'DisPredHeteroDCRNN2'
            model.module.adj_dis = adj_dis
            model.module.adj_precip = adj_precip
            model.module.adj_precip2dis = adj_precip2dis

            optim = torch.optim.Adam([
                {'params': model.module.mdcrnn.parameters(), 'lr': lr},
                {'params': model.module.dense_readout.parameters(), 'lr': lr},
                {'params': model.module.proj_dis.parameters(), 'lr': lr},
                {'params': model.module.proj_precip.parameters(), 'lr': lr},
                {'params': model.module.aggr_nodes.parameters(), 'lr': lr},
            ])

        else:
            model.adj_dis = adj_dis
            model.adj_precip = adj_precip
            model.adj_precip2dis = adj_precip2dis
            model.name = 'DisPredHeteroDCRNN2'

            optim = torch.optim.Adam([
                {'params': model.mdcrnn.parameters(), 'lr': lr},
                {'params': model.dense_readout.parameters(), 'lr': lr},
                {'params': model.proj_dis.parameters(), 'lr': lr},
                {'params': model.proj_precip.parameters(), 'lr': lr},
                {'params': model.aggr_nodes.parameters(), 'lr': lr},
            ])

        _, model = mo.train(device,
                            train_x, train_y, val_x, val_y,
                            target_in_forward,
                            model, train_loop, val_loop, batch_size, lr,
                            optim=optim)

        pred = mo.pred_4_test_hedcrnn(model, test_x, target_in_forward, device)
        pred = pred[:, 0, :]
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

        # field discharge
        test_df_field.index = test_df_field.index.ceil('H')
        test_df_field = test_df_field.groupby(level=0).mean()
        test_df['field'] = test_df_field['discharge']
        test_df = test_df[~test_df['field'].isna()]

    return test_df
