import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


def normed_mae(true, pred, quantile=1):
    mae = mean_absolute_error(true, pred)
    nmae = mae / true.quantile(quantile)
    return nmae


def metric_bias(true, pred):
    return np.sum((true - pred)) / true.shape[0]


def metric_nse(true, pred):
    numerator = mean_squared_error(true, pred)
    denominator = mean_squared_error(true, np.full(true.shape, np.mean(true)))
    return 1 - (numerator/ denominator)

def metric_cc(true, pred):
    numerator = np.sum((pred - np.mean(pred)) * (true - np.mean(true)))
    denominator_1 = mean_squared_error(pred, np.full(pred.shape, np.mean(pred)), squared=False) * np.sqrt(pred.shape[0])
    denominator_2 = mean_squared_error(true, np.full(true.shape, np.mean(true)), squared=False) * np.sqrt(true.shape[0])
    return numerator / (denominator_1 * denominator_2)


def metric_peak(gage, modeled_pred_high):
    flooding_period = pd.read_csv(f'./outputs/USGS_{gage}/flooding_period/action_period.csv')
    flooding_period['start'] = pd.to_datetime(flooding_period['start'])
    flooding_period['end'] = pd.to_datetime(flooding_period['end'])
    flooding_period['peak_time'] = pd.to_datetime(flooding_period['peak_time'])

    peak_dis_per_diff_list, peak_dis_diff_list, peak_time_diff_list = [], [], []
    for index, row in flooding_period.iterrows():
        start_time = row['start']
        end_time = row['end']
        peak_dis = row['peak_dis']
        peak_time = row['peak_time']
        data_avail = row['data_avail']

        if data_avail:
            one_modeled_pred_high = modeled_pred_high[
                (modeled_pred_high.index >= start_time) & (modeled_pred_high.index <= end_time)
            ]
            pred_peak_dis = one_modeled_pred_high.max()
            pred_peak_time = one_modeled_pred_high[one_modeled_pred_high == pred_peak_dis].index[0]

            peak_dis_diff_list.append((pred_peak_dis - peak_dis))
            peak_dis_per_diff_list.append((pred_peak_dis - peak_dis) / peak_dis)
            peak_time_diff_list.append((pred_peak_time - peak_time).total_seconds() / 3600)

    peak_dis_mae = sum([abs(i) for i in peak_dis_diff_list]) / len(peak_dis_diff_list)
    peak_dis_mape = sum([abs(i) for i in peak_dis_per_diff_list]) / len(peak_dis_per_diff_list)
    peak_time_bias = sum(peak_time_diff_list) / len(peak_time_diff_list)
    return peak_dis_mae, peak_dis_mape, peak_time_bias


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


def metric_dis_pred_report(df, df_full, threshold, gage):
    report_dict = {}
    metric_names = ['MAE', 'MAPE', 'RMSE', 'BIAS', 'NSE', 'CC']

    df = df[~df['pred'].isna()]
    df_full = df_full[~df_full[['modeled', 'pred']].isna().any(axis=1)]

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
        mean_absolute_percentage_error(field_true, field_pred),
        mean_squared_error(field_true, field_pred, squared=False),
        metric_bias(field_true, field_pred),
        metric_nse(field_true, field_pred),
        metric_cc(field_true, field_pred),
    ]
    report_dict['field_high'] = [
        mean_absolute_error(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
        mean_absolute_percentage_error(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
        mean_squared_error(field_true_high, field_pred_high, squared=False) if len(field_true_high) != 0 else np.nan,
        metric_bias(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
        metric_nse(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
        metric_cc(field_true_high, field_pred_high) if len(field_true_high) != 0 else np.nan,
    ]
    report_dict['modeled_overall'] = [
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_absolute_error(modeled_true, modeled_pred),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_absolute_percentage_error(modeled_true, modeled_pred),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_squared_error(modeled_true, modeled_pred, squared=False),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_bias(modeled_true, modeled_pred),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_nse(modeled_true, modeled_pred),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_cc(modeled_true, modeled_pred),
    ]
    report_dict['modeled_overall_high'] = [
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_absolute_error(modeled_true_high, modeled_pred_high),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_absolute_percentage_error(modeled_true_high, modeled_pred_high),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else mean_squared_error(modeled_true_high, modeled_pred_high, squared=False),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_bias(modeled_true_high, modeled_pred_high),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_nse(modeled_true_high, modeled_pred_high),
        np.nan if (modeled_pred.isna().any()) | (len(modeled_pred) == 0) else metric_cc(modeled_true_high, modeled_pred_high),
    ]
    # report_dict['modeled_overall_high'] = [
    #     np.nan if modeled_pred.isna().any() else mean_absolute_error(modeled_true_high, modeled_pred_high),
    #     np.nan if modeled_pred.isna().any() else mean_absolute_percentage_error(modeled_true_high, modeled_pred_high),
    #     np.nan if modeled_pred.isna().any() else mean_squared_error(modeled_true_high, modeled_pred_high, squared=False),
    #     np.nan if modeled_pred.isna().any() else metric_bias(modeled_true_high, modeled_pred_high),
    #     np.nan if modeled_pred.isna().any() else metric_nse(modeled_true_high, modeled_pred_high),
    #     np.nan if modeled_pred.isna().any() else metric_cc(modeled_true_high, modeled_pred_high),
    # ]

    if (gage == '01573560') & (len(modeled_pred) > 100):
        add_peak_metric = ['PEAK_BIAS', 'PEAK_PER_BIAS', 'PEAK_TIME_BIAS']
        metric_names = metric_names + add_peak_metric
        report_dict['modeled_overall_high'] = report_dict['modeled_overall_high'] + list(metric_peak(
            gage, modeled_pred_high
        ))
        for k, v in report_dict.items():
            if k != 'modeled_overall_high':
                report_dict[k] = v + [np.nan] * len(add_peak_metric)

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
