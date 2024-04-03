import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn, Tensor
import optuna
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.conv import MessagePassing

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

import warnings

expr_dir_global = ''
best_score_optuna_tune = 1e10


class DConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(1, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def _degree_norm(edge_index, edge_weight, device):

        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight)
        adj_mat = adj_mat.reshape(adj_mat.size(1), adj_mat.size(2))

        deg_in = torch.matmul(
            torch.ones(size=(1, adj_mat.size(0))).to(device), adj_mat
        )
        deg_in = deg_in.flatten()
        deg_in_inv = torch.reciprocal(deg_in)

        row, col = edge_index
        norm_in = deg_in_inv[col]
        return norm_in

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
        edge_index_2: torch.LongTensor = None,
        edge_weight_2: torch.FloatTensor = None,
    ) -> Tensor:

        device = X.device

        if edge_index_2 is not None and edge_weight_2 is not None:
            norm_in_2 = self._degree_norm(edge_index_2, edge_weight_2, device)
            norm_in = self._degree_norm(edge_index, edge_weight, device)

            # 1st message (target nodes themselves have been excluded)
            message_1_i = self.propagate(
                edge_index_2, x=X, norm=norm_in_2, size=(
                    torch.max(edge_index).item() + 1, # number of p nodes
                    torch.max(edge_index_2[1, :]).item() + 1  # number of s nodes
                )
            )
            h = torch.matmul(message_1_i, self.weight[0][1])

            # 2nd message
            message_2_1_i = self.propagate(edge_index, x=X, norm=norm_in, size=None)
            message_2_2_i = self.propagate(
                edge_index_2, x=message_2_1_i, norm=norm_in_2,
                size=(
                    torch.max(edge_index).item() + 1, # number of p nodes
                    torch.max(edge_index_2[1, :]).item() + 1  # number of s nodes
                )
            )
            h += torch.matmul(message_2_2_i, self.weight[0][2])

            # iterate from 3rd
            for k in range(3, self.weight.size(1)):
                message_3_1_i = self.propagate(edge_index, x=message_2_1_i, norm=norm_in, size=None)
                message_3_2_i = self.propagate(
                    edge_index_2, x=message_3_1_i, norm=norm_in_2,
                    size=(
                        torch.max(edge_index).item() + 1, # number of p nodes
                        torch.max(edge_index_2[1, :]).item() + 1  # number of s nodes
                    )
                )
                h += torch.matmul(message_3_2_i, self.weight[0][k])
                message_2_1_i = message_3_1_i

        else:
            norm_in = self._degree_norm(edge_index, edge_weight, device)
            h = torch.matmul(X, self.weight[0][0])

            message_1_i = None
            if self.weight.size(1) > 1:
                message_1_i = self.propagate(edge_index, x=X, norm=norm_in, size=None)
                h += torch.matmul(message_1_i, self.weight[0][1])

            for k in range(2, self.weight.size(1)):
                message_2_i = self.propagate(edge_index, x=message_1_i, norm=norm_in, size=None)
                h += torch.matmul(message_2_i, self.weight[0][k])
                message_1_i = message_2_i

        if self.bias is not None:
            h += self.bias

        return h


class SingleDCRNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
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

    def _calculate_update_gate_enhanced(self, X_s, edge_index_s, edge_weight_s, H):
        Z_s = torch.cat([X_s, H[:, :, :]], dim=-1)
        Z_s = self.conv_x_z(Z_s, edge_index_s, edge_weight_s)
        Z = torch.sigmoid(Z_s)
        return Z

    def _calculate_reset_gate_enhanced(self, X_s, edge_index_s, edge_weight_s, H,):
        R_s = torch.cat([X_s, H[:, :, :]], dim=-1)
        R_s = self.conv_x_r(R_s, edge_index_s, edge_weight_s)
        R = torch.sigmoid(R_s)
        return R

    def _calculate_candidate_state_enhanced(self, X_s, edge_index_s, edge_weight_s, H, R):
        gated_H = H * R
        H_tilde_s = torch.cat([X_s, gated_H[:, :, :]], dim=-1)
        H_tilde_s = self.conv_x_h(H_tilde_s, edge_index_s, edge_weight_s)
        H_tilde = torch.tanh(H_tilde_s)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X_s: torch.FloatTensor,
        edge_index_s: torch.LongTensor,
        edge_weight_s: torch.FloatTensor,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X_s[:, :, 0], H)
        Z = self._calculate_update_gate_enhanced(X_s, edge_index_s, edge_weight_s, H,)
        R = self._calculate_reset_gate_enhanced(X_s, edge_index_s, edge_weight_s, H,)
        H_tilde = self._calculate_candidate_state_enhanced(X_s, edge_index_s, edge_weight_s, H, R,)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class HomoDCRNN(nn.Module):
    def __init__(
            self,
            feat_in,
            K, feat_out_dcrnn,
            num_readout_layer, num_node,
            num_forward=1
    ):
        super().__init__()
        self.sdcrnn = SingleDCRNN(feat_in, feat_out_dcrnn, K, bias=True)
        self.gru = nn.GRU(feat_in, feat_out_dcrnn, batch_first=True)
        self.num_forward = num_forward
        self.dense_readout = mo.stack_dense_layers(feat_out_dcrnn * 2, num_forward * 1, num_readout_layer)
        self.num_node = num_node

    def forward(self, x):

        # data prep
        x_precip = x[:, :, self.num_node:]
        x_dis = x[:, :, :self.num_node]
        x_dis = x_dis.unsqueeze(-1)

        edge_index_dis = np.vstack(np.where(self.adj_dis.to_numpy() != 0))
        edge_index_dis = torch.from_numpy(edge_index_dis).to(torch.long).to(x_dis.device)
        edge_weight_dis = self.adj_dis.to_numpy()[np.where(self.adj_dis != 0)[0], np.where(self.adj_dis != 0)[1]]
        edge_weight_dis = torch.from_numpy(edge_weight_dis).float().to(x_dis.device)

        # diffusion convolution rnn
        h_dis = None
        for i in range(x_dis.shape[1]):
            h_dis = self.sdcrnn(x_dis[:, i, :, :], edge_index_dis, edge_weight_dis, h_dis)

        # precip embedding
        o, h_precip = self.gru(x_precip)
        h_precip = h_precip[0, :, :, ].unsqueeze(1)
        # h_precip = torch.cat([h_precip] * self.num_node, dim=1)

        # readout
        # h = torch.concat((h_dis, h_precip), dim=-1)
        h = torch.concat((h_dis[:, 0:1, :], h_precip), dim=-1)
        out = self.dense_readout(h)

        return out


def train_loop(dataloader, model, optimizer, device):

    loss_func_1 = mo.WeightedMSELoss()
    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device, dtype=torch.float), y.to(device)

        # y_target = y[:, 0, 5:10].to(dtype=torch.float)  # hard coded
        # y_target_weights = y[:, 0, :5].to(dtype=torch.float)  # hard coded
        y_target = y[:, 0, 5:6].to(dtype=torch.float)  # hard coded
        y_target_weights = y[:, 0, 0:1].to(dtype=torch.float)  # hard coded

        pred = model(x)
        pred = pred[:, :, 0]
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

        # y_target = y[:, 0, 5:10].to(dtype=torch.float)
        y_target = y[:, 0, 5:6].to(dtype=torch.float)
        # y_level_weights = y[:, :, 0].to(dtype=torch.float)
        pred = model(x)
        pred = pred[:, :, 0]

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
        feat_in, k,
        adj_dis,
        target_in_forward, num_nodes, num_rep=1
):

    batch_size = trial.suggest_int("batch_size", low=128, high=256, step=64)
    lr = trial.suggest_float("lr", low=0.0003, high=0.0013, step=0.0002)
    feat_out = trial.suggest_int("feat_out", low=16, high=80, step=32)
    layer_out = trial.suggest_int("layer_out", low=2, high=4, step=1)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = HomoDCRNN(feat_in, k, feat_out, layer_out, num_nodes)

        model.adj_dis = adj_dis
        model.name = 'DisPredHomoDCRNN5'
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
    if objective_value < best_score_optuna_tune:
        best_score_optuna_tune = objective_value
        torch.save(
            {
                'model': model,
                'optimizer': optim.state_dict(),
            },
            f'{expr_dir_global}/best_HODCRNN_optuna_tune_{objective_value}.pth'
        )
    return objective_value


def train_pred(
        df, df_precip, df_field, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, if_tune
):

    global expr_dir_global
    expr_dir_global = expr_dir

    # parameters - tune
    n_trials = 50
    tune_rep_num = 1

    # # parameters - default model, lead time 1
    # batch_size = 192
    # lr = 0.0009
    # feat_out = 80
    # layer_out = 2
    #
    # # parameters - default model, lead time 2
    # batch_size = 192
    # lr = 0.0013
    # feat_out = 80
    # layer_out = 2

    # parameters - default model, lead time 3
    batch_size = 192
    lr = 0.0009
    feat_out = 80
    layer_out = 4

    # # parameters - default model, lead time 4
    # batch_size = 192
    # lr = 0.0009
    # feat_out = 48
    # layer_out = 2
    #
    # # parameters - default model, lead time 5
    # batch_size = 128
    # lr = 0.0007
    # feat_out = 80
    # layer_out = 4


    k = 3
    feat_in = 1
    num_nodes = 5
    scaler_power = PowerTransformer

    # other parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    adj_dis = pd.read_csv(f'{adj_matrix_dir}/adj_matrix.csv', index_col=0)

    df = df.resample('H', closed='right', label='right').mean()
    scaler_stream = scaler_power()
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
    scaler_precip = scaler_power()
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
        tag = 'hodcrnn'
        study = optuna.create_study(direction="minimize",
                                    storage=f"sqlite:///tuner/db_dis_pred_{tag}.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_hodcrnn.sqlite3 --port 8083
                                    study_name=f"dis_pred_{tag}_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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

        # model = HomoDCRNN(feat_in, k, feat_out, layer_out, num_nodes)

        saved = torch.load('./outputs/experiments/hodcrnn_5__2024-03-13-12-48-00/best_HODCRNN_optuna_tune_0.029401084408164024.pth')
        model = saved['model']
        model.eval()
        model.to(device)
        model.name = 'DisPredHomoDCRNN_tune'

        # model.name = 'DisPredHomoDCRNN5'
        # model.adj_dis = adj_dis
        # optim = torch.optim.Adam([
        #     {'params': model.sdcrnn.parameters(), 'lr': lr},
        #     {'params': model.dense_readout.parameters(), 'lr': lr},
        #     {'params': model.gru.parameters(), 'lr': lr},
        # ])
        #
        # _, model = mo.train(device,
        #                     train_x, train_y, val_x, val_y,
        #                     target_in_forward,
        #                     model, train_loop, val_loop, batch_size, lr,
        #                     optim=optim)
        # torch.save(model, f'{expr_dir}/model.pth')

        pred = mo.pred_4_test_hodcrnn(model, test_x, target_in_forward, device)
        pred = pred[:, 0, :]
        scaler_pred = scaler_power()
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
