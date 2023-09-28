import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import optuna

import utils.features as ft
import models.LSTM as lstm


class PILSTM(nn.Module):
    def __init__(self, num_feature, num_forward, num_class, num_lstm_layers, size_lstm,
                 num_dense_layers_class, num_dense_layers_level_discharge, num_dense_layers_stem):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=num_feature, hidden_size=size_lstm, num_layers=num_lstm_layers, batch_first=True)
        self.num_forward = num_forward
        self.num_class = num_class

        # auto generate dense layer, given the number of dense layers
        self.dense_class = self.stack_dense_layers(size_lstm * num_lstm_layers, num_forward * num_class, num_dense_layers_class)
        # NOTE: 2 is hard coded here, it means water level and discharge
        self.dense_level_discharge = self.stack_dense_layers(size_lstm * num_lstm_layers, num_forward * 2,
                                                             num_dense_layers_level_discharge)

        # NOTE: 2 is hard coded here, it means water level and discharge
        self.dense_stem = self.stack_dense_layers(size_lstm * num_lstm_layers, size_lstm * num_lstm_layers,
                                                  num_dense_layers_stem)
        self.dense_stem_no_branch = self.stack_dense_layers(size_lstm * num_lstm_layers, num_forward * 2,
                                                            num_dense_layers_stem)
        self.dense_class_no_branch = self.stack_dense_layers(num_forward * 2 + 2, num_forward * num_class,
                                                             num_dense_layers_class)

    def forward(self, x):
        _, (hn, cn) = self.LSTM(x)

        # class logit
        hn_to_dense = torch.cat([hn[i] for i in range(hn.size(0))], dim=-1)

        if self.dense_level_discharge:
            if self.dense_stem:
                stem_to_branch = self.dense_stem(hn_to_dense)
                out_class = self.dense_class(stem_to_branch)
                out_level_discharge = self.dense_level_discharge(stem_to_branch)
            else:
                out_class = self.dense_class(hn_to_dense)
                out_level_discharge = self.dense_level_discharge(hn_to_dense)
        else:
            if self.dense_stem:
                stem = self.dense_stem_no_branch(hn_to_dense)
                out_level_discharge = stem

                x_last_time = x[:, -1, 1:]
                out_class = self.dense_class_no_branch(torch.cat((stem, x_last_time), -1))
            else:
                raise ValueError("Stem dense layers cannot be empty when the water dense layer is empty.")

        out_class = out_class.view(-1, self.num_class, self.num_forward)
        # NOTE: 2 is hard coded here, it means water level and discharge
        out_level_discharge = out_level_discharge.view(-1, self.num_forward, 2)

        return out_class, out_level_discharge

    def stack_dense_layers(self, input_size, output_size, num_layers):
        # USE: make a stack of dense layers, given the number of layers

        if num_layers == 0:
            return None
        else:
            dense_size_step = math.floor((input_size - output_size) / num_layers)
            dense_size_list = [input_size - i * dense_size_step for i in range(num_layers)]
            dense_size_list.append(output_size)
            dense_layers = []
            for i in range(num_layers):
                dense_layers.append(nn.Linear(dense_size_list[i], dense_size_list[i + 1]))
            dense_layers = nn.Sequential(*dense_layers)
            return dense_layers


def train_loop(dataloader, model, loss_func, weight_loss_level_discharge, optimizer, device):

    loss_func_2 = nn.MSELoss()

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device, dtype=torch.float), y.to(device)

        y_class = y[:, :, 0].to(dtype=torch.int64)
        y_level_discharge = y[:, :, 1:].to(dtype=torch.float)

        pred_class, pred_level_discharge = model(x)

        loss_class = loss_func(pred_class, y_class)
        loss_level_discharge = loss_func_2(pred_level_discharge, y_level_discharge)

        loss = loss_class + loss_level_discharge * weight_loss_level_discharge

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss: >7f}  [{current: >5d} / {size: >5d}]")


def val_loop_legacy(dataloader, model, target_in_forward, loss_func, weight_loss_level_discharge, device):

    loss_func_2 = nn.MSELoss()

    model.eval()
    val_loss = 0
    val_loss_f1 = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, dtype=torch.float), y.to(device)

            y_class = y[:, :, 0].to(dtype=torch.int64)
            y_level_discharge = y[:, :, 1:].to(dtype=torch.float)

            pred_class, pred_level_discharge = model(x)
            loss_class = loss_func(pred_class, y_class).item()
            loss_level_discharge = loss_func_2(pred_level_discharge, y_level_discharge).item()

            val_loss += loss_class + loss_level_discharge * weight_loss_level_discharge

            softmax_func = nn.Softmax(dim=1)
            pred_class = torch.argmax(softmax_func(pred_class), dim=1).cpu().numpy()
            pred_class = pred_class[:, [0]]
            val_loss_f1 += f1_score(np.squeeze(y_class[:, target_in_forward - 1].cpu().numpy()), np.squeeze(pred_class),
                                    zero_division=0)

    val_loss /= len(dataloader)
    print(f"Avg loss: {val_loss:>8f}")

    val_loss_f1 /= len(dataloader)
    print(f"Avg F1 for positive class: {val_loss_f1:>8f} \n")

    return val_loss, -val_loss_f1


def val_loop(dataloader, model, target_in_forward, loss_func, weight_loss_level_discharge, device):

    loss_func_2 = nn.MSELoss()

    model.eval()

    with torch.no_grad():
        x_list = []
        y_list = []
        for x, y in dataloader:
            x_list.append(x)
            y_list.append(y)
            x = torch.cat(x_list, dim=0).to(device, dtype=torch.float)
            y = torch.cat(y_list, dim=0).to(device)

        y_class = y[:, :, 0].to(dtype=torch.int64)
        y_level_discharge = y[:, :, 1:].to(dtype=torch.float)

        pred_class, pred_level_discharge = model(x)
        loss_class = loss_func(pred_class, y_class).item()
        loss_level_discharge = loss_func_2(pred_level_discharge, y_level_discharge).item()

        val_loss = loss_class + loss_level_discharge * weight_loss_level_discharge

        softmax_func = nn.Softmax(dim=1)
        pred_class = torch.argmax(softmax_func(pred_class), dim=1).cpu().numpy()
        pred_class = pred_class[:, [0]]
        val_loss_f1 = f1_score(np.squeeze(y_class[:, target_in_forward - 1].cpu().numpy()), np.squeeze(pred_class),
                                zero_division='warn')

    print(f"Avg loss: {val_loss:>8f}")
    print(f"Avg F1 for positive class: {val_loss_f1:>8f} \n")

    return val_loss, -val_loss_f1


def train(device,
          df, train_x, train_y, val_x, val_y,
          target, forward, target_in_forward,
          if_weight,
          batch_size, lr,
          weight_loss_ld,
          num_lstm_layers,
          size_lstm,
          num_dense_layers_class,
          num_dense_layers_ld,
          cross_entropy_label_smoothing,
          num_dense_layers_stem,
          trial=None,
          ):
    # USE: computation with hyperparameters for optuna tuning
    #      the computation from dataloader to obtaining trained model
    # INPUT and OUTPUT: same as train_pred()

    train_dataloader = DataLoader(lstm.RiverDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(lstm.RiverDataset(val_x, val_y), batch_size=batch_size, shuffle=True)

    if if_weight:
        num_class = df[target].nunique()
        weights = len(df) / (num_class * np.bincount(df[target].values.astype(int)))
        weight_tensor = torch.tensor(weights.tolist()).to(device)
    else:
        weight_tensor = None

    # model
    model = PILSTM(train_x.shape[2], len(forward), 2,
                   num_lstm_layers=num_lstm_layers,
                   size_lstm=size_lstm,
                   num_dense_layers_class=num_dense_layers_class,
                   num_dense_layers_level_discharge=num_dense_layers_ld,
                   num_dense_layers_stem=num_dense_layers_stem,
                   )
    model.to(device)

    # train
    print("Training begins.")
    if weight_tensor is not None:
        loss_func = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=cross_entropy_label_smoothing)
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=cross_entropy_label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 500
    es = lstm.EarlyStopper(patience=5, min_delta=0)
    best_val_metric = 1e8

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_func, weight_loss_ld, optimizer, device)
        val_loss, val_metric = val_loop(val_dataloader, model, target_in_forward, loss_func, weight_loss_ld, device)

        es.stopper(val_loss)
        if es.stop:
            break

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), 'best_model.pth')

        if trial:
            trial.report(best_val_metric, t)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.load_state_dict(torch.load('best_model.pth'))
    os.remove('best_model.pth')

    print("Training done.")

    return best_val_metric, model


def train_w_hp(trial, device, df, train_x, train_y, val_x, val_y, target, forward, target_in_forward, if_weight,
               num_rep=1):
    # USE: computation with hyperparameters for optuna tuning
    #      the computation from dataloader to obtaining trained model
    # INPUT and OUTPUT: same as train_pred()

    batch_size = trial.suggest_int("batch_size", low=64, high=448, step=64)
    lr = trial.suggest_float("lr", low=0.00001, high=0.01, step=0.000999)

    num_lstm_layers = trial.suggest_int("lstm_layer", low=1, high=3, step=1)
    size_lstm = trial.suggest_int("lstm_size", low=128, high=1024, step=128)

    num_dense_layers_class = trial.suggest_int("dense_layer_class", low=1, high=5, step=1)
    num_dense_layers_ld = trial.suggest_int("dense_layer_ld", low=0, high=5, step=1)

    # if there is ld branch, there must be at lease one layer of stem dense
    if num_dense_layers_ld == 0:
        num_dense_layers_stem = trial.suggest_int("dense_layer_stem", low=1, high=5, step=1)
    else:
        num_dense_layers_stem = trial.suggest_int("dense_layer_stem", low=0, high=5, step=1)

    weight_loss_ld = trial.suggest_float("weight_loss_ld", low=0, high=4, step=0.5)

    cross_entropy_label_smoothing = trial.suggest_float("cross_entropy_label_smoothing", low=0, high=0.02, step=0.001)

    best_val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)
        best_val_metric, _ = train(device, df, train_x, train_y, val_x, val_y,
                                   target, forward, target_in_forward, if_weight, batch_size, lr, weight_loss_ld,
                                   num_lstm_layers=num_lstm_layers,
                                   size_lstm=size_lstm,
                                   num_dense_layers_class=num_dense_layers_class,
                                   num_dense_layers_ld=num_dense_layers_ld,
                                   cross_entropy_label_smoothing=cross_entropy_label_smoothing,
                                   num_dense_layers_stem=num_dense_layers_stem,
                                   trial=trial,)
        best_val_metric_ave += best_val_metric
    return best_val_metric_ave / num_rep


def train_pred(df,
               features, target, lags, forward,
               target_in_forward,
               val_percent, test_percent,
               if_weight, batch_size, lr,
               weight_loss_ld,
               if_tune, tune_rep_num,
               if_cv,
               ):
    # USE: use standard LSTM to pred water level surge
    # INPUT: df, pandas df
    #        features, list of str, col names in df
    #        target, str, col names in df
    #        lags, forward, list of ints, starting from 1
    #        target_in_forward, the position of target in forward list, starting from 1
    #        val_percent, test_percent, float, 0-1
    #        if_weight, if weight targets
    # OUTPUT: df, one col is true, another is pred

    # parameters - device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parameters - cv
    num_fold_cv = 3
    test_percent_cv = 0.05
    val_percent_cv = 0.15

    # parameters - tune
    n_trials = 200

    # parameters - default model
    num_lstm_layers = 2
    size_lstm = 128
    num_dense_layers_class = 2
    num_dense_layers_ld = 3
    num_dense_layers_stem = 0
    cross_entropy_label_smoothing = 0.002

    # data
    df = (df - df.min()) / (df.max() - df.min())
    df['index'] = range(len(df))
    sequences_w_index = ft.create_sequences(df, lags, forward, features + ['index'], target)

    # conserve tune switch
    if_tune_original = if_tune

    # prepare for index split
    dataset_index = []
    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]

    # get the index at the sample dim without cv
    if not if_cv:
        num_fold = 2

        tscv = TimeSeriesSplit(num_fold, test_size=math.floor(x.shape[0] * test_percent))
        for i, (train_index, test_index) in enumerate(tscv.split(x)):
            train_index, val_index, _, _ = train_test_split(train_index, train_index,
                                                            test_size=val_percent / (1 - (num_fold - i) * test_percent),
                                                            shuffle=False)
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Val: index={val_index}")
            print(f"  Test:  index={test_index}")
            dataset_index.append({'train_index': train_index, 'val_index': val_index, 'test_index': test_index})

        dataset_index = dataset_index[-1:]
        print("Only keep the fold at the end of the time series.")

    # get the index at the sample dim with cv
    else:
        tscv = TimeSeriesSplit(num_fold_cv, test_size=math.floor(x.shape[0] * test_percent_cv))
        for i, (train_index, test_index) in enumerate(tscv.split(x)):
            train_index, val_index, _, _ = train_test_split(train_index, train_index,
                                                            test_size=val_percent_cv / (
                                                                        1 - (num_fold_cv - i) * test_percent_cv),
                                                            shuffle=False)
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Val: index={val_index}")
            print(f"  Test:  index={test_index}")
            dataset_index.append({'train_index': train_index, 'val_index': val_index, 'test_index': test_index})

    # use the index to split data
    test_df_list = []
    i = 0

    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]
    y = sequences_w_index[:, :, :-1][:, -len(forward):, :]
    y_index = sequences_w_index[:, :, [-1]][:, -len(forward):, :]
    for fold_index in dataset_index:
        print(f"Train and pred of Fold {i}: ")
        train_x = x[fold_index['train_index'], :, :]
        train_y = y[fold_index['train_index'], :]
        val_x = x[fold_index['val_index'], :, :]
        val_y = y[fold_index['val_index'], :, :]
        test_x = x[fold_index['test_index'], :, :]
        test_y = y[fold_index['test_index'], :, 0]
        test_y_index = y_index[fold_index['test_index'], :, 0]

        # train with hp tuning
        if if_tune:
            study = optuna.create_study(direction="minimize",
                                        storage="sqlite:///tuner/db.sqlite3",
                                        # optuna-dashboard sqlite:///tuner/db.sqlite3
                                        study_name="pilstm_" + 'cv-' + str(i) + '_' +
                                                   datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                        )
            study.optimize(lambda trial: train_w_hp(trial, device,
                                                    df, train_x, train_y, val_x, val_y,
                                                    target, forward, target_in_forward, if_weight, tune_rep_num),
                           n_trials=n_trials,
                           )

            # get the best hps and close if_tune
            best_hps = study.best_trial.params
            batch_size = best_hps['batch_size']
            lr = best_hps['lr']
            num_lstm_layers = best_hps['lstm_layer']
            size_lstm = best_hps['lstm_size']
            num_dense_layers_class = best_hps['dense_layer_class']
            num_dense_layers_ld = best_hps['dense_layer_ld']
            num_dense_layers_stem = best_hps['dense_layer_stem']
            weight_loss_ld = best_hps['weight_loss_ld']
            cross_entropy_label_smoothing = best_hps['cross_entropy_label_smoothing']

            # disable tune for training model using best hp
            if_tune = False

        # train without hp tuning
        if not if_tune:
            _, model = train(device, df,
                             train_x, train_y, val_x, val_y,
                             target, forward, target_in_forward, if_weight, batch_size, lr, weight_loss_ld,
                             num_lstm_layers, size_lstm, num_dense_layers_class, num_dense_layers_ld,
                             cross_entropy_label_smoothing, num_dense_layers_stem)

            # pred
            test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
            model.eval()
            softmax_func = nn.Softmax(dim=1)
            prob_y = softmax_func(model(test_x_tensor)[0])
            pred_y = torch.argmax(prob_y, dim=1).cpu().numpy()
            pred_y = pred_y[:, target_in_forward - 1]

            # store results
            test_y_datatime_index = df.iloc[test_y_index[:, target_in_forward - 1]].index
            test_df = pd.DataFrame(test_y[:, target_in_forward - 1], columns=['true'], index=test_y_datatime_index)
            test_df['pred'] = pred_y

            # store the results of folds
            test_df_list.append(test_df)

            # turn on tune switch if original if_tune is true:
            if if_tune_original:
                if_tune = True

        i += 1

    test_df = pd.concat(test_df_list)
    return test_df
