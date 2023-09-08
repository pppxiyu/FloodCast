import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import utils.features as ft


class river_dataset(Dataset):
    def __init__(self, x, y):
        self.X = np.copy(x)
        self.Y = np.copy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        feature = self.X[i]
        target = self.Y[i]
        return feature, target


class LSTM(nn.Module):
    def __init__(self, num_feature, num_forward, num_class):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=num_feature, hidden_size=512, num_layers=1, batch_first=True)
        self.dense = nn.Linear(512, num_forward * num_class)
        self.num_forward = num_forward
        self.num_class = num_class

    def forward(self, x):
        _, (hn, cn) = self.LSTM(x)
        out = self.dense(hn)
        return out.view(-1, self.num_class, self.num_forward)


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


def train_loop(dataloader, model, loss_func, optimizer, device):

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.int64)

        pred = model(x)
        loss = loss_func(pred, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss: >7f}  [{current: >5d} / {size: >5d}]")


def val_loop(dataloader, model, loss_func, device):

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.int64)
            pred = model(x)
            val_loss += loss_func(pred, y).item()

    val_loss /= len(dataloader)
    print(f"Avg loss: {val_loss:>8f} \n")

    return val_loss


def train_pred(df,
               features, target, lags, forward,
               target_in_forward,
               val_percent, test_percent,
               if_weight,
               batch_size,
               learning_rate,
               random_seed,
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

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # datasets
    df = (df - df.min()) / (df.max() - df.min())
    df['index'] = range(len(df))
    sequences_w_index = ft.create_sequences(df, lags, forward, features + ['index'], target)
    train_x, train_y, val_x, val_y, test_x, test_y = ft.split_sequences(
        sequences_w_index[:, :, :-1],
        val_percent, test_percent,
        forward,
        random_seed,
        shuffle=False)
    _, _, _, _, _, test_y_w_index = ft.split_sequences(
        sequences_w_index[:, :, [-1]], val_percent, test_percent, forward, random_seed, shuffle=False)

    train_dataloader = DataLoader(river_dataset(train_x, train_y.astype(int)), batch_size=batch_size)
    val_dataloader = DataLoader(river_dataset(val_x, val_y.astype(int)), batch_size=batch_size)

    if if_weight:
        num_class = df[target].nunique()
        weights = len(df) / (num_class * np.bincount(df[target].values.astype(int)))
        weight_tensor = torch.tensor(weights.tolist()).to(device)
    else:
        weight_tensor = None

    # model
    model = LSTM(train_x.shape[2], len(forward), 2)
    model.to(device)

    # train
    if weight_tensor is not None:
        loss_func = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 500
    es = EarlyStopper(patience=5, min_delta=0)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_func, optimizer, device)
        val_loss = val_loop(val_dataloader, model, loss_func, device)
        es.stopper(val_loss)
        if es.stop:
            break
    print("Training done.")

    # pred
    test_x_tensor = torch.tensor(test_x).to(device, dtype=torch.float)
    model.eval()
    softmax_func = nn.Softmax(dim=1)
    prob_y = softmax_func(model(test_x_tensor))
    pred_y = torch.argmax(prob_y, dim=1).cpu().numpy()
    pred_y = pred_y[:, target_in_forward - 1]

    # store results
    test_y_datatime_index = df.iloc[test_y_w_index[:, target_in_forward - 1]].index
    test_df = pd.DataFrame(test_y[:, target_in_forward - 1], columns=['true'], index=test_y_datatime_index)
    test_df['pred'] = pred_y

    return test_df
