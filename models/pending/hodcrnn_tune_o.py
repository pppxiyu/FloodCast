import numpy as np
import torch

from sklearn.preprocessing import PowerTransformer

import utils.features as ft
import utils.modeling as mo
import utils.preprocess as pp
from utils.features import process_tune_data


def train_pred(
        df, df_precip, df_field, adj_matrix_dir,
        lags, forward, target_gage,
        val_percent, test_percent, expr_dir, if_tune
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load base model
    saved = torch.load('./outputs/USGS_01573560/best_dis_HODCRNN_optuna_tune_0.0002539197448641062.pth')
    model = saved['model']
    model.to(device)
    model.name = 'DisPredHomoDCRNN_tune'
    model.dense_flex_1 = None
    model.dense_flex_2 = None

    # parameters
    model_res_name = 'xgboost'

    if model_res_name == 'xgboost':
        num_sampling = 50
        num_rep = 40

    # data
    df = df.resample('H', closed='right', label='right').mean()
    df_dis_normed = (df - df.min()) / (df.max() - df.min())
    dis_cols = [col for col in df.columns if col.endswith('00060')]
    df_dis_normed = df_dis_normed[dis_cols]
    for col in df_dis_normed:
        if col.endswith('00060'):
            df_dis_normed = pp.sample_weights(df_dis_normed, col, if_log=True)

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

    # process
    train_x_raw, val_x_raw, test_x, test_y_index, train_df_field, val_df_field, test_df_field = process_tune_data(
        df_field, df_dis_normed,
        sequences_w_index,
        val_percent, test_percent,
        forward,
    )

    # data for tuning: use model out as x
    train_x_pred_o = model(
        torch.tensor(train_x_raw).to(device, dtype=torch.float)
    ).detach().cpu().numpy()[:, 0:1, 0].astype('float64')
    val_x_pred_o = model(
        torch.tensor(val_x_raw).to(device, dtype=torch.float)
    ).detach().cpu().numpy()[:, 0:1, 0].astype('float64')

    train_x_pred_o = (train_x_pred_o * (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min())
                    + df[f"{target_gage}_00060"].min())
    val_x_pred_o = (val_x_pred_o * (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min())
                    + df[f"{target_gage}_00060"].min())

    transformer_x_o = PowerTransformer(method='yeo-johnson')
    transformer_x_o.fit(df[f"{target_gage}_00060"].values[:len(df) // 2, np.newaxis])

    train_x_tune_o = transformer_x_o.transform(train_x_pred_o)
    val_x_tune_o = transformer_x_o.transform(val_x_pred_o)
    # transformer_x.inverse_transform(train_x_tune)

    # data for tuning: use field measure as y
    train_y_field = train_df_field['discharge'].values[:, np.newaxis].astype(np.float64)
    val_y_field = val_df_field['discharge'].values[:, np.newaxis].astype(np.float64)

    train_y_res = train_y_field - train_x_pred_o
    val_y_res = val_y_field - val_x_pred_o

    transformer_y = PowerTransformer(method='yeo-johnson')
    transformer_y.fit(np.concatenate(
        (train_y_res, val_y_res),
        axis=0)
    )
    train_y_tune = np.concatenate((
        train_df_field['discharge_weights'].values[:, np.newaxis],
        transformer_y.transform(train_y_res)),
        axis=1,
    )
    val_y_tune = np.concatenate((
        val_df_field['discharge_weights'].values[:, np.newaxis],
        transformer_y.transform(val_y_res)),
        axis=1,
    )

    # test set
    test_x_pred = mo.pred_4_test_hodcrnn(model, test_x, target_in_forward, device)
    test_x_pred = test_x_pred[:, 0, :].astype(np.float64)
    test_x_pred = (test_x_pred * (df[f"{target_gage}_00060"].max() - df[f"{target_gage}_00060"].min())
                   + df[f"{target_gage}_00060"].min())
    test_x_tune = transformer_x_o.transform(test_x_pred)

    # residual error learning
    if model_res_name == 'linear':
        from sklearn.linear_model import LinearRegression
        model_res = LinearRegression()
        model_res.fit(
            np.concatenate((train_x_tune_o, val_x_tune_o), axis=0),
            np.concatenate((train_y_tune[:, 1:], val_y_tune[:, 1:]), axis=0),
            sample_weight=np.concatenate((train_y_tune[:, 0], val_y_tune[:, 0]), axis=0)
        )
        residual_pred = model_res.predict(test_x_tune)

    elif model_res_name == 'xgboost':
        from models.baselines.xgboost import XGBRegressor

        residual_pred = np.zeros(test_x_tune.shape)
        for i in range(num_rep):
            indices = np.arange(train_x_tune_o.shape[0] + val_x_tune_o.shape[0])
            np.random.shuffle(indices)
            model_res = XGBRegressor(
                n_estimators=500,
                objective='reg:squarederror',
                # early_stopping_rounds=5,
                eval_metric='mae',
                gamma=0,
                # max_depth=max_depth, learning_rate=lr, reg_alpha=reg_alpha,
            )
            model_res.fit(
                np.concatenate((
                    train_x_tune_o,
                    val_x_tune_o
                ), axis=0)[indices[:num_sampling]],
                np.concatenate((
                    train_y_tune[:, 1:],
                    val_y_tune[:, 1:]
                ), axis=0)[indices[:num_sampling]],
                # eval_set = [
                #     (train_x_tune, train_y_tune[:, 1:]),
                #     (val_x_tune, val_y_tune[:, 1:])
                # ],
                sample_weight=np.concatenate((
                    train_y_tune[:, 0],
                    val_y_tune[:, 0]
                ), axis=0)[indices[:num_sampling]],
            )
            residual_pred += model_res.predict(test_x_tune)[:, np.newaxis]
        residual_pred = residual_pred / num_rep

    residual_pred = transformer_y.inverse_transform(residual_pred)
    pred_y_tune = residual_pred + test_x_pred

    # modeled discharge
    test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
    test_df = test_df.rename(columns={
        f'{target_gage}_00060': 'modeled',
        f'{target_gage}_00065': 'water_level',
    })

    # pred discharge
    test_df['pred_w_o_res'] = test_x_pred
    test_df['pred_res'] = residual_pred
    test_df['pred'] = pred_y_tune
    test_df_full = test_df.copy()

    # field discharge
    test_df_field.index = test_df_field.index.ceil('H')
    test_df_field = test_df_field.groupby(level=0).mean()
    test_df['field'] = test_df_field['discharge']
    test_df = test_df[~test_df['field'].isna()]

    return test_df, test_df_full
