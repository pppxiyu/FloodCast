import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def normed_mae(true, pred, quantile=1):
    mae = mean_absolute_error(true, pred)
    nmae = mae / true.quantile(quantile)
    return nmae


def metric_classi_report(df):
    # USE: print classification report
    # INPUT: test_results df
    # OUTPUT:
    true = df['true']
    pred = df['pred']
    target_names = ['No flooding', 'Flooding']
    report_dict = classification_report(true, pred, target_names=target_names, output_dict=True)
    print(classification_report(true, pred, target_names=target_names))

    return report_dict


def metric_reg_report(df, nmae_quantile=0.85):
    true = df['true']
    pred = df['pred']
    mae = mean_absolute_error(true, pred)
    nmae = normed_mae(true, pred, nmae_quantile)

    print(f'MAE is: {mae}, MAE normed by quantile {nmae_quantile} value is {nmae}.')

    report_dict = {'MAE': mae, 'NMAE': nmae}
    return report_dict


def metric_dis_pred_tune_report(df):
    field = df['field']
    pred_wo_tune = df['pred_w_o_tune']
    pred_tune = df['pred_tune']

    mae_field_wo = mean_absolute_error(field, pred_wo_tune)
    mape_field_wo = mean_absolute_percentage_error(field, pred_wo_tune)

    mae_field = mean_absolute_error(field, pred_tune)
    mape_field = mean_absolute_percentage_error(field, pred_tune)

    print(f'MAE (field) (wo tune) is: {mae_field_wo}, '
          f'MAPE (field) (wo tune) is: {mape_field_wo}')
    print(f'MAE (field) (tune) is: {mae_field}, '
          f'MAPE (field) (tune) is: {mape_field}')

    report_dict = {'MAE (field) (wo tune)': mae_field_wo, 'MAPE (field) (wo tune)': mape_field_wo,
                   'MAE (field) (tune)': mae_field, 'MAPE (field) (tune)': mape_field}

    return report_dict


def metric_dis_pred_report(df, df_full, threshold):
    report_dict = {}
    metric_names = ['MAE', 'MAPE']

    df = df[~df['pred'].isna()]

    field_true = df['field']
    field_pred = df['pred']
    modeled_true = df_full['modeled']
    modeled_pred = df_full['pred']
    field_true_high = df[df['water_level'] >= threshold]['field']
    field_pred_high = df[df['water_level'] >= threshold]['pred']
    modeled_true_high = df_full[df_full['water_level'] >= threshold]['modeled']
    modeled_pred_high = df_full[df_full['water_level'] >= threshold]['pred']

    report_dict['field_overall'] = [
        mean_absolute_error(field_true, field_pred),
        mean_absolute_percentage_error(field_true, field_pred)
    ]
    report_dict['field_high'] = [
        mean_absolute_error(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
        mean_absolute_percentage_error(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan
    ]
    report_dict['modeled_overall'] = [
        np.nan if modeled_pred.isna().any() else mean_absolute_error(modeled_true, modeled_pred),
        np.nan if modeled_pred.isna().any() else mean_absolute_percentage_error(modeled_true, modeled_pred)
    ]
    report_dict['modeled_overall_high'] = [
        np.nan if modeled_pred.isna().any() else mean_absolute_error(modeled_true_high, modeled_pred_high),
        np.nan if modeled_pred.isna().any() else  mean_absolute_percentage_error(modeled_true_high, modeled_pred_high)
    ]

    report_df = pd.DataFrame(report_dict)
    report_df.index = metric_names

    print(report_df)
    return report_df


def metric_dis_pred_report_legacy(df, df_full, model_name, threshold):
    if model_name in [
        'naive', 'linear', 'gru', 'mlp', 'xgboost',
        'hodcrnn', 'hedcrnn', 'pi_hedcrnn_tune_1', 'pilstm_tune_half',
    ]:
        report_dict = metric_dis_pred_report(df, df_full, threshold)
    elif model_name in ['hedcrnn_tune', 'pi_hedcrnn_tune_2', 'lstm_tune']:
        report_dict = metric_dis_pred_tune_report(df)

    return report_dict
