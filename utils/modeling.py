import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.init as init

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import optuna
import os
import math


class RiverDataset(Dataset):
    def __init__(self, x, y):
        self.X = np.copy(x)
        self.Y = np.copy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        feature = self.X[i]
        target = self.Y[i]
        return feature, target


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.stop = False

    def stopper(self, val_loss):
        if val_loss < (self.min_val_loss - self.min_delta):
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def create_weighted_cross_entropy(target_series, if_weight, cross_entropy_label_smoothing, device, reduction='mean'):
    if if_weight:
        num_class = target_series.nunique()
        weights = len(target_series) / (num_class * np.bincount(target_series.values.astype(int)))
        weight_tensor = torch.tensor(weights.tolist()).to(device)
    else:
        weight_tensor = None

    if weight_tensor is not None:
        loss_func = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=cross_entropy_label_smoothing, reduction=reduction)
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=cross_entropy_label_smoothing, reduction=reduction)

    return loss_func


def stack_dense_layers(input_size, output_size, num_layers, bias=True, activation=nn.ReLU()):
    # USE: make a stack of dense layers, given the number of layers

    if num_layers == 0:
        return None
    else:
        dense_size_step = math.floor((input_size - output_size) / num_layers)
        dense_size_list = [input_size - i * dense_size_step for i in range(num_layers)]
        dense_size_list.append(output_size)
        dense_layers = []
        for i in range(num_layers):
            dense_layers.append(nn.Linear(dense_size_list[i], dense_size_list[i + 1], bias=bias))
            if i != list(range(num_layers))[-1]:
                dense_layers.append(activation)
        dense_layers = nn.Sequential(*dense_layers)
        return dense_layers


def stack_dense_layers_rating_curve(num_layers, max_dim, num_features=1, num_out=1,
                                    activation=nn.ReLU(), invert=False):
    # USE: make a stack of dense layers, given the number of layers

    num_layers += 1  # consider the inputs also as a layer, just for convenience.

    if num_layers == 2:
        layers = []
        layers.append(nn.Linear(num_features, num_out))
        layers.append(activation)
        return nn.Sequential(*layers)

    elif num_layers > 2:
        num_layers_first_half = num_layers // 2
        num_layers_second_half = num_layers - num_layers_first_half - 1

        step_first_half = (max_dim - num_features) / num_layers_first_half
        step_second_half = (max_dim - num_out) / num_layers_second_half

        neuron_sequence = [num_features]
        for i in range(num_layers_first_half):
            i += 1
            neuron_sequence.append(math.ceil(num_features + step_first_half * i))
        peak = neuron_sequence[-1]
        for i in range(num_layers_second_half - 1):
            i += 1
            neuron_sequence.append(math.ceil(peak - step_second_half * i))
        neuron_sequence.append(1)

        dense_layers = []
        for i in range(len(neuron_sequence) - 1):
            if invert:
                dense_layers.append(activation)
                dense_layers.append(nn.Linear(neuron_sequence[i], neuron_sequence[i + 1]))
            else:
                dense_layers.append(nn.Linear(neuron_sequence[i], neuron_sequence[i + 1]))
                dense_layers.append(activation)
        dense_layers = nn.Sequential(*dense_layers)
        return dense_layers

    else:
        raise ValueError("The layer count is too small to form a net.")


def train(device,
          train_x, train_y, val_x, val_y,
          target_in_forward,
          model, train_loop, val_loop,
          batch_size, lr,
          optim=None,
          trial=None,
          remove_best_model=True,
          add_val_loss_func=None,
          ):
    # USE: computation with hyperparameters for optuna tuning
    #      the computation from dataloader to obtaining trained model
    # INPUT and OUTPUT: same as train_pred()

    print("Training begins.")

    train_dataloader = DataLoader(RiverDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(RiverDataset(val_x, val_y), batch_size=batch_size, shuffle=True)

    model.to(device)
    model_name = model.name

    # weighted_cross_entropy = create_weighted_cross_entropy(df[target[0]], if_weight, cross_entropy_label_smoothing,
    #                                                        device)

    if optim:
        optimizer = optim
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 500
    es = EarlyStopper(patience=7, min_delta=0)
    best_val_metric = 1e8

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(
            train_dataloader, model, optimizer, device,
        )
        val_loss, val_metric = val_loop(val_dataloader, train_dataloader,
                                        model, target_in_forward,
                                        device)
        if add_val_loss_func:
            val_loss, val_metric = val_loop(val_dataloader, train_dataloader,
                                            model, target_in_forward,
                                            device, add_loss_func=add_val_loss_func)

        es.stopper(val_loss)
        if es.stop:
            break

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')

        if trial:
            trial.report(best_val_metric, t)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))

    torch.save(model, f'best_model_{model_name}.pth')
    if remove_best_model:
        os.remove(f'best_model_{model_name}.pth')

    print("Training done.")

    return best_val_metric, model


def train_loop_rc(dataloader, model, optimizer, device, if_weight, train_x=None, train_y=None):
    loss_func_mse = nn.MSELoss()
    loss_func_weighted_mse = WeightedMSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

        pred = model(x)
        if if_weight:
            loss = loss_func_weighted_mse(torch.squeeze(pred), y, y[:, 0, 1].unsqueeze(1).unsqueeze(2))
        else:
            loss = loss_func_mse(pred, y)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
                    print(f"Gradient of {name} is NaN")

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss: >8f}  [{current: >5d} / {size: >5d}]")

            # wl, dis = rc_run(model, 0.6775, 14.045, 0, 6525.0, 0.01)
            # plt.scatter((train_x * (14.045 - 0.6775)) + 0.6775, (train_y * (6525.0 - 0)) + 0, color='orange',
            #             label='given', s=0.5)
            # plt.scatter(wl, dis, color='blue', label='fitted', s=0.01)
            # plt.legend()
            # plt.show()

            print('Vis!')


def val_loop_rc(dataloader, model, device, if_weight):
    loss_func_mse = nn.MSELoss()
    loss_func_weighted_mse = WeightedMSELoss()

    model.eval()

    with torch.no_grad():
        x_list = []
        y_list = []
        for x, y in dataloader:
            x_list.append(x)
            y_list.append(y)
        x = torch.cat(x_list, dim=0).to(device, dtype=torch.float)
        y = torch.cat(y_list, dim=0).to(device, dtype=torch.float)

        pred = torch.squeeze(model(x))

        if if_weight:
            val_loss = loss_func_weighted_mse(pred, y[:, 0], y[:, 0, 1].unsqueeze(1).unsqueeze(2))
        else:
            val_loss = loss_func_mse(pred, y[:, 0]).item()

        val_metric = loss_func_mse(pred, y[:, 0]).item()

    print(f"Avg loss: {val_loss:>9f}")
    print(f"Avg val metric: {val_metric:>9f}")

    return val_loss, val_metric


def train_rc(device, train_x, train_y, val_x, val_y, model,
             batch_size, lr, direction, rc_type='rc', if_weight=False, trial=None):
    print("RC pretraining begins.")

    train_dataloader = DataLoader(RiverDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(RiverDataset(val_x, val_y), batch_size=batch_size, shuffle=True)

    model.to(device)
    model_name = model.name

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 5000
    es = EarlyStopper(patience=5, min_delta=0)
    best_val_metric = 1e8

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop_rc(train_dataloader, model, optimizer, device, if_weight, train_x, train_y)
        val_loss, val_metric = val_loop_rc(val_dataloader, model, device, if_weight)

        es.stopper(val_loss)
        if es.stop:
            break

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model, f'best_model_{model_name}.pth')

        if trial:
            trial.report(best_val_metric, t)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    best_val_metric_in_training = best_val_metric
    model = torch.load(f'best_model_{model_name}.pth')
    os.remove(f'best_model_{model_name}.pth')

    # best rc in this training is assigned to model
    # check if previously saved one is better
    model_updated = False
    for filename in os.listdir('.'):
        if filename.startswith(f'saved_best_{rc_type}_{direction}_') and filename.endswith('.pth'):
            saved_metric = filename.replace(f'saved_best_{rc_type}_{direction}_', '')
            saved_metric = float(saved_metric.replace('.pth', ''))
            if saved_metric < best_val_metric:
                saved_model = torch.load(filename)
                if direction == 'direct':
                    if hasattr(model, 'a'):
                        model.a = saved_model.a
                        model.b = saved_model.b
                        model.c = saved_model.c
                    elif hasattr(model, 'rating_curve'):
                        model.rating_curve = saved_model.rating_curve
                elif direction == 'inverse':
                    if hasattr(model, 'a'):
                        model.a = saved_model.a
                        model.b = saved_model.b
                        model.c = saved_model.c
                    elif hasattr(model, 'rating_curve_inv'):
                        model.rating_curve_inv = saved_model.rating_curve_inv
                else:
                    raise ValueError('Direction param was not specified.')
                model_updated = True
                best_val_metric = saved_metric

    # if the rc in this training remains the best one
    if model_updated == False:
        for filename in os.listdir('.'):
            if filename.startswith(f'saved_best_{rc_type}_{direction}_') and filename.endswith('.pth'):
                os.remove(filename)
        best_val_metric_save = f"{best_val_metric:.9f}"
        torch.save(model, f'saved_best_{rc_type}_{direction}_{best_val_metric_save}.pth')

    print("Training done.")

    return best_val_metric_in_training, model


def pred_4_test(model, test_x, target_in_forward, device):
    # pred
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()
    softmax_func = nn.Softmax(dim=1)

    if isinstance(model(test_x_tensor), tuple):
        model_output = model(test_x_tensor)[0]
    else:
        model_output = model(test_x_tensor)

    prob_y = softmax_func(model_output)
    pred_y = torch.argmax(prob_y, dim=1).cpu().numpy()
    pred_y = pred_y[:, target_in_forward - 1]

    return pred_y


def pred_4_test_reg_legacy(model, test_x, target_in_forward, device):
    # pred
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()

    if isinstance(model(test_x_tensor), tuple):
        model_output = model(test_x_tensor)[0]
    else:
        model_output = model(test_x_tensor)

    pred_y = model_output.cpu().detach().numpy()
    pred_y = pred_y[:, target_in_forward - 1]

    return pred_y


def pred_4_test_hedcrnn(model, test_x, target_in_forward, device, batch_size=2048):
    # pred
    model.eval()

    output_list = []
    for i in range(0, test_x.shape[0], batch_size):
        test_x_tensor = torch.tensor(
            test_x[i: min(i + batch_size, test_x.shape[0])]
        ).to(device, dtype=torch.float)

        x_dis = test_x_tensor[:, :, :5].unsqueeze(-1)  # hard coded here
        x_precip = test_x_tensor[:, :, 5:].unsqueeze(-1)  # hard coded here

        model_output  = model(x_dis, x_precip).cpu().detach().numpy()
        output_list.append(model_output)
    pred_y = np.concatenate(output_list, axis=0)

    return pred_y


def pred_4_test_hodcrnn(model, test_x, target_in_forward, device, batch_size=2048):
    # pred
    model.eval()

    output_list = []
    for i in range(0, test_x.shape[0], batch_size):
        test_x_tensor = torch.tensor(
            test_x[i: min(i + batch_size, test_x.shape[0])]
        ).to(device, dtype=torch.float)

        model_output = model(test_x_tensor).cpu().detach().numpy()
        output_list.append(model_output)
    pred_y = np.concatenate(output_list, axis=0)

    return pred_y


def pred_4_test_hodcrnn_tune_h(model, encoder, test_x, target_in_forward, device, batch_size=2048):
    # pred
    model.eval()

    output_list = []
    for i in range(0, test_x.shape[0], batch_size):
        test_x_tensor = torch.tensor(
            test_x[i: min(i + batch_size, test_x.shape[0])]
        ).to(device, dtype=torch.float)

        model_output = encoder(model, device, test_x_tensor).cpu().detach().numpy()
        output_list.append(model_output)
    pred_y = np.concatenate(output_list, axis=0)

    return pred_y

def pred_4_test_dis(model, test_x, target_in_forward, device):
    # pred
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()

    if isinstance(model(test_x_tensor), tuple):
        model_output = model(test_x_tensor)[0]
    else:
        model_output = model(test_x_tensor)

    model_output = model.rc_direct(model_output)
    pred_y = model_output.cpu().detach().numpy()
    pred_y = pred_y[:, target_in_forward - 1]

    return pred_y


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, weights):
        if not (inputs.shape == targets.shape == weights.shape):
            raise ValueError("Input, Target, and Weights must be of the same shape")

        squared_diffs = (inputs - targets) ** 2
        loss = weights * squared_diffs

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss


class RatingCurveSoftInv(nn.Module):
    def __init__(self, neuron, num_layers):
        super().__init__()

        self.inv = False
        self.rating_curve = stack_dense_layers_rating_curve(num_layers, neuron)
        self.rating_curve_inv = stack_dense_layers_rating_curve(num_layers, neuron)

    def direct(self):
        self.inv = False
        return

    def inverse(self):
        self.inv = True
        return

    def forward(self, x):
        if self.inv:
            return self.rating_curve_inv(x)
        else:
            return self.rating_curve(x)


class RatingCurvePhysical(nn.Module):
    def __init__(self):
        super().__init__()

        self.inv = None
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU()

        init.uniform_(self.a, 0.1, 1)
        init.uniform_(self.c, 1, 1.5)
        init.uniform_(self.b, 0.1, 1)

    def direct(self):
        self.inv = False
        return

    def inverse(self):
        self.inv = True
        return

    def forward(self, x):
        if self.inv:
            out = ((x / self.a) ** (1 / self.c)) - self.relu(self.b)
            return out
        else:
            out = self.a * ((x + self.relu(self.b)) ** self.c)
            return out


class RatingCurvePhysicalTemporal(nn.Module):
    def __init__(self, max_dim, num_layers):
        super().__init__()
        self.inv = False
        self.a = stack_dense_layers_rating_curve(num_layers, max_dim, num_features=2)  # h_{t-1} and h_t
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

        init.uniform_(self.c, 0.1, 1)
        init.uniform_(self.b, 0.01, 0.1)

    def forward(self, x):
        out = self.a(x) * (nn.ReLU()(x[:, 1:2] + self.b) ** self.c)
        return out


class RatingCurvePhysicalAdapted(nn.Module):
    def __init__(self, max_dim_a, num_layers_a, max_dim_b, num_layers_b):
        super().__init__()
        self.inv = False
        self.a = stack_dense_layers_rating_curve(num_layers_a, max_dim_a, num_features=1)  # h_t
        self.c = stack_dense_layers_rating_curve(num_layers_b, max_dim_b, num_features=1)  # h_t

        self.b = nn.Parameter(torch.randn(1))
        # self.c = nn.Parameter(torch.randn(1))
        init.constant_(self.b, 0.1)
        # init.constant_(self.c, inits[2])

    def forward(self, x):
        out = self.a(x) * ((x + self.b) ** self.c(x))
        return out


def power_function(x, a, b, c):
    return a * np.power((x + b), c)


def check_training_f1_per_epoch(train_dataloader, indices, model, device, target_in_forward):
    with torch.no_grad():
        x = train_dataloader.dataset[indices][0]
        y = train_dataloader.dataset[indices][1]
        x_tensor = torch.tensor(x).to(device, dtype=torch.float)

        y_class = torch.tensor(y[:, :, 0]).to(device, dtype=torch.int64)
        pred_class = model(x_tensor)[0]

        softmax_func = nn.Softmax(dim=1)
        pred_class = torch.argmax(softmax_func(pred_class), dim=1).cpu().numpy()
        pred_class = pred_class[:, [0]]
        train_f1 = f1_score(np.squeeze(y_class[:, target_in_forward - 1].cpu().numpy()), np.squeeze(pred_class),
                            zero_division='warn')

    return train_f1


def check_training_mae_per_epoch(train_dataloader, indices, model, device, target_in_forward):
    loss_func = torch.nn.L1Loss()

    with torch.no_grad():
        x = train_dataloader.dataset[indices][0]
        y = train_dataloader.dataset[indices][1]
        x_tensor = torch.tensor(x).to(device, dtype=torch.float)

        y_ld = torch.tensor(y[:, :, 0]).to(device, dtype=torch.int64)
        pred_ld = model(x_tensor)[0]

        train_mae = loss_func(pred_ld.squeeze(), y_ld.squeeze())

    return train_mae.item()


def calculate_split_indices(num_samples, num_splits):
    indices = list(range(num_samples))
    split_size = num_samples // num_splits
    remainder = num_samples % num_splits
    split_indices = []

    start = 0
    for i in range(num_splits):
        end = start + split_size + (1 if i < remainder else 0)
        split_indices.append(indices[start:end])
        start = end

    return split_indices


def check_rc_compliance_pilstm_free_form_soft_hybrid(test_x, device, model, denorm=True):
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()

    last_water_level = test_x[:, -1, 1]
    water_level_diff = model(test_x_tensor)[1].cpu().detach().numpy()[:, 0, 0]
    pred_water_level = water_level_diff + last_water_level

    last_dis = test_x[:, -1, 3]
    dis_diff = model(test_x_tensor)[2].cpu().detach().numpy()[:, 0, 0]
    pred_discharge = dis_diff + last_dis

    if denorm:
        pred_water_level = (pred_water_level * (33.12 - 15.55)) + 15.55
        pred_discharge = (pred_discharge * (72100.0 - 590.0)) + 590

    return pred_water_level, pred_discharge


def check_rc_compliance_pilstm_free_form_soft_multiO(test_x, device, model, denorm=True):
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()

    last_water_level = test_x[:, -1, 1]
    water_level_diff = model(test_x_tensor)[1].cpu().detach().numpy()[:, 0, 0]
    pred_water_level = water_level_diff + last_water_level
    pred_discharge = model(test_x_tensor)[2].cpu().detach().numpy()[:, 0, 0]

    if denorm:
        pred_water_level = (pred_water_level * (33.12 - 15.55)) + 15.55
        pred_discharge = (pred_discharge * (72100.0 - 590.0)) + 590

    return pred_water_level, pred_discharge


def rc_run(model, x_min=15.55, x_max=33.12, y_min=590, y_max=72100, step=0.1):
    x = np.arange(x_min, x_max, step)
    x_normed = ((x - x_min) / (x_max - x_min)).tolist()
    x_tensor = torch.tensor(x_normed).reshape(len(x_normed), 1, 1).to(next(model.parameters()).device)
    y_tensor = model(x_tensor)
    y_out = y_tensor.cpu().detach().numpy()
    y_out = (y_out * (y_max - y_min)) + y_min
    return x, np.squeeze(y_out)


def rc_inv_run(model_inv):
    x = np.arange(590.0, 72100.0, 100)
    x_normed = ((x - 590) / (72100.0 - 590.0)).tolist()
    x_tensor = torch.tensor(x_normed).reshape(len(x_normed), 1, 1).to(next(model_inv.parameters()).device)
    y_tensor = model_inv(x_tensor)
    y_out = y_tensor.cpu().detach().numpy()
    y_out = (y_out * (33.12 - 15.55)) + 15.55
    return x, np.squeeze(y_out)


def trc_run(model, df):

    x = df[['water_level_lag', 'water_level']].values
    x_out = (x * (33.12 - 15.55)) + 15.55

    x_tensor = torch.tensor(x).to(next(model.parameters()).device).float()
    y_tensor = model(x_tensor)
    y_out = y_tensor.cpu().detach().numpy()
    y_out = (y_out * (72100.0 - 590.0)) + 590.0
    return x_out, np.squeeze(y_out)


def rc_run_w_residual(model, model_residual_rc, x_min=15.55, x_max=33.12, y_min=590, y_max=72100, step=0.1):
    x = np.arange(x_min, x_max, step)
    x_normed = ((x - x_min) / (x_max - x_min)).tolist()
    x_tensor = torch.tensor(x_normed).reshape(len(x_normed), 1, 1).to(next(model.parameters()).device)
    y_tensor_normed = model(x_tensor)
    y_tensor = (y_tensor_normed * (y_max - y_min)) + y_min

    y_tensor_residual = model_residual_rc(x_tensor)
    y_out = (y_tensor + y_tensor_residual).cpu().detach().numpy()
    # y_out = (y_out * (y_max - y_min)) + y_min
    return x, np.squeeze(y_out)


def approx_rc(row, df, buffer_len=2, time_col_name='date_time', level_col_name='01573560_00065'):

    df_wl_col = [c for c in df.columns if c.endswith('_00065')]
    assert len(df_wl_col) == 1, 'Multi or zero water level col.'
    df_dis_col = [c for c in df.columns if c.endswith('_00060')]
    assert len(df_dis_col) == 1, 'Multi or zero discharge col.'

    t_upper_bound = row[time_col_name] + pd.DateOffset(days=buffer_len//2)
    t_lower_bound = row[time_col_name] - pd.DateOffset(days=buffer_len//2)
    df_buffer = df[(df.index > t_lower_bound) & (df.index < t_upper_bound)]
    df_buffer['diff'] = df_buffer[df_wl_col[0]] - row[level_col_name]
    df_diff = df_buffer.abs().sort_values(by=['diff'])
    if df_diff.iloc[0]['diff'] < 1e-5:
        return df_diff.iloc[0][df_dis_col[0]]
    else:
        print(f'Bad luck! Exactly same water level within {buffer_len} days buffer is not found. Check it!')
        neighbors = df_diff.iloc[0:2]
        if neighbors[df_wl_col[0]].nunique() == 2:
            estimate = (
                    (neighbors[df_dis_col[0]].iloc[1] - neighbors[df_dis_col[0]].iloc[0]) *
                    (row[level_col_name] - neighbors[df_wl_col[0]].iloc[0]) /
                    (neighbors[df_wl_col[0]].iloc[1] - neighbors[df_wl_col[0]].iloc[0]) +
                    neighbors[df_dis_col[0]].iloc[0]
            )
        else:
            i = 1
            while neighbors[df_wl_col[0]].nunique() == 1:
                neighbors = df_diff.iloc[0:2 + i]
                i += 1
            neighbors = neighbors.drop_duplicates(subset=df_wl_col[0])
            estimate = (
                    (neighbors[df_dis_col[0]].iloc[1] - neighbors[df_dis_col[0]].iloc[0]) *
                    (row[level_col_name] - neighbors[df_wl_col[0]].iloc[0]) /
                    (neighbors[df_wl_col[0]].iloc[1] - neighbors[df_wl_col[0]].iloc[0]) +
                    neighbors[df_dis_col[0]].iloc[0]
            )
        return estimate


def convert_array_w_rc(pred_wl, df_w_time, df_raw):
    time_col_name = df_w_time.index.name if df_w_time.index.name is not None else 'index'
    df_w_time['pred_level'] = pred_wl
    df_w_time = df_w_time.reset_index()
    df_w_time['pred_dis'] = df_w_time.apply(
        lambda row: approx_rc(
            row,
            df_raw[['01573560_00065', '01573560_00060']],
            buffer_len=2,
            time_col_name=time_col_name, level_col_name='pred_level'
        ), axis=1
    )
    pred_dis = df_w_time['pred_dis'].values
    return pred_dis

