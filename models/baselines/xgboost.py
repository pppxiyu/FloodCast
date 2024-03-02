import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
import optuna
import utils.features as ft
import utils.preprocess as pp


def train_w_hp(
        trial, device, train_x, train_y, val_x, val_y,
        target_in_forward, num_rep=1
):

    lr = trial.suggest_float("lr", low=0.8, high=1.2, step=0.1)
    max_depth = trial.suggest_int("max_depth", low=9, high=9.1, step=4)
    reg_alpha = trial.suggest_int("reg_alpha", low=0, high=.1, step=4)

    val_metric_ave = 0
    for i in range(num_rep):
        print("Repeat: ", i)

        model = XGBRegressor(
            n_estimators=500,
            objective='reg:squarederror',
            early_stopping_rounds=7, eval_metric='mae',
            gamma=0, max_depth=max_depth, learning_rate=lr, reg_alpha=reg_alpha,
            device=device,
        )
        model.fit(
            train_x.reshape(train_x.shape[0], -1),
            train_y[:, 0, 5:],  # hard coded
            eval_set = [
                (train_x.reshape(train_x.shape[0], -1), train_y[:, 0, 5:]),
                (val_x.reshape(val_x.shape[0], -1), val_y[:, 0, 5:])
            ],
            sample_weight=train_y[:, 0, :5].mean(1))

        val_metric_ave += model.best_score
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
    lr = 0.8
    reg_alpha = 0
    max_depth = 9
    device = "cuda"

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
    test_percent_updated, test_df_field, num_test_sequences = ft.update_test_percent(
        df_field, df_dis_normed,
        sequences_w_index, test_percent
    )
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
        print('flag')
        study = optuna.create_study(direction="minimize",
                                    storage=f"sqlite:///tuner/db_dis_pred_xgb.sqlite3",
                                    # optuna-dashboard sqlite:///tuner/db_dis_pred_xgb.sqlite3 --port 8081
                                    study_name=f"dis_pred_xgb_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        study.optimize(
            lambda trial: train_w_hp(
                trial, device,
                train_x, train_y, val_x, val_y,
                target_in_forward, tune_rep_num),
            n_trials=n_trials,
        )

        # get the best hps and close if_tune
        best_hps = study.best_trial.params
        lr = best_hps['lr']
        max_depth = best_hps['max_depth']
        reg_alpha = best_hps['reg_alpha']

        # disable tune for training model using best hp
        if_tune = False

    # train without hp tuning
    if not if_tune:

        model = XGBRegressor(
            n_estimators=500,
            objective='reg:squarederror',
            early_stopping_rounds=7, eval_metric='mae',
            gamma=0, max_depth=max_depth, learning_rate=lr, reg_alpha=reg_alpha,
            device=device,
        )
        model.fit(
            train_x.reshape(train_x.shape[0], -1),
            train_y[:, 0, 5:],  # hard coded
            eval_set = [
                (train_x.reshape(train_x.shape[0], -1), train_y[:, 0, 5:]),
                (val_x.reshape(val_x.shape[0], -1), val_y[:, 0, 5:])
            ],
            sample_weight=train_y[:, 0, :5].mean(1))

        pred = model.predict(test_x.reshape(test_x.shape[0], -1))
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
