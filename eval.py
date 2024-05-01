import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as io
import torch
import os
import utils.preprocess as pp
import json

analysis_name = 'table_metrics'


if analysis_name == 'scatter_accuracy':
    dir_expr = './outputs/experiments'
    dir_save_fig = './papers/figs'
    horizon = 1
    model_group_label = 'shallow'
    models = [
        # 'naive',
        'linear', 'mlp', 'xgboost',
        # 'gru', 'hodcrnn', 'pi_hodcrnn',
    ]
    models_label = {
        # 'naive': 'Persistence',
        'linear': 'Linear',
        'mlp': 'MLP',
        'xgboost': 'XGBoost',
        # 'gru': 'GRU',
        # 'hodcrnn': 'DCRNN',
        # 'pi_hodcrnn': 'DCRNN+RC',
    }
    models_symbol = {
        # 'naive': 'diamond',
        'linear': 'circle',
        'mlp': 'triangle-up',
        'xgboost': 'cross',
        # 'gru': 'circle',
        # 'hodcrnn': 'triangle-up',
        # 'pi_hodcrnn': 'cross',
    }
    models_color = {
        # 'naive': '#F27B35',
        'linear': '#386084',
        'mlp': '#BF4163',
        'xgboost': '#75556B'
        # 'gru': '#992f87',
        # 'hodcrnn': '#552e81',
        # 'pi_hodcrnn': '#efae42'
    }

    # import data
    all_expr = [e for e in os.listdir(dir_expr) if e.startswith('ARCHIVE')]

    df_dict = {}
    for model in models:
        expr_names = [e for e in all_expr if e.startswith(f'ARCHIVE_{model}_{horizon}__')]
        expr_names.sort(reverse=True)
        expr_name = expr_names[0]
        df = pd.read_csv(f'{dir_expr}/{expr_name}/test_df_full.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        df_dict[model] = df

    # vis
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(27000)), y=list(range(27000)),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    for k, v in df_dict.items():
        fig.add_trace(go.Scatter(
            x=v['modeled'].to_list(), y=v['pred'].to_list(),
            mode='markers',
            marker=dict(symbol=models_symbol[k], color=models_color[k]),
            name=models_label[k]),
        )
    fig.update_layout(
        template='seaborn',
        yaxis_title='Forecasted streamflow (feet\u00B3)',
        xaxis_title='Reported streamflow (feet\u00B3)',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            range=[0, 27000], tick0=0, dtick=5000,
            ticks='outside', tickmode='linear',
            tickformat=',',
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            range=[0, 27000], tick0=0, dtick=5000,
            ticks='outside', tickmode='linear',
            tickformat=',',
        ),
        width=375, height=375,
        legend=dict(x=0.01, y=0.99,),
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="center",
        #     x=0.5
        # ),
    )
    fig.write_html(f'{dir_save_fig}/scatter_report_vs_pred_{horizon}_{model_group_label}.html')
    io.write_image(fig, f'{dir_save_fig}/scatter_report_vs_pred_{horizon}_{model_group_label}.png', scale=4)

if analysis_name == 'line_flood':

    target_gage = '01573560'
    dir_expr = './outputs/experiments'
    dir_save_fig = './papers/figs'
    horizon = 6
    models = ['gru', 'hodcrnn', 'pi_hodcrnn']
    models_label = {
        'gru': 'GRU',
        'hodcrnn': 'DCRNN',
        'pi_hodcrnn': 'DCRNN+RC'
    }
    models_symbol = {
        'gru': 'dash',
        'hodcrnn': 'dot',
        'pi_hodcrnn': 'dashdot'
    }
    models_color = {
        'gru': '#992f87',
        'hodcrnn': '#552e81',
        'pi_hodcrnn': '#efae42'
    }
    tz = 'America/New_York'

    dir_flood_period = './outputs/USGS_01573560/flooding_period'
    action_period = pd.read_csv(f'{dir_flood_period}/action_period.csv')
    action_period = action_period[action_period['data_avail'] == True]
    action_period['start'] = pd.to_datetime(action_period['start'], utc=True).dt.tz_convert(tz)
    action_period['end'] = pd.to_datetime(action_period['end'], utc=True).dt.tz_convert(tz)
    action_num = 3
    buffer = 44

    # import data
    all_expr = [e for e in os.listdir(dir_expr) if e.startswith('ARCHIVE')]
    df_dict = {}
    for model in models:
        expr_names = [e for e in all_expr if e.startswith(f'ARCHIVE_{model}_{horizon}__')]
        expr_names.sort(reverse=True)
        expr_name = expr_names[0]
        df = pd.read_csv(f'{dir_expr}/{expr_name}/test_df_full.csv', index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
        df = df[
            (df.index >= (action_period.iloc[action_num]['start'] - pd.Timedelta(buffer, 'h')))
            & (df.index <= (action_period.iloc[action_num]['end'] + pd.Timedelta(buffer, 'h')))
            ]
        df_dict[model] = df

    # import precip data
    data_precip = pp.import_data_precipitation_legacy(
        f'./data/JAXA_precipitation_data/USGS_{target_gage}',
        ['40.2', '40.3', '40.4', '40.5', '40.6'],
        ['-76.2', '-76.3', '-76.4', '-76.5', '-76.6', '-76.7', '-76.8'],
        'America/New_York'
    )
    adj_matrix_dir = f'./outputs/USGS_{target_gage}/adj_matrix_USGS_{target_gage}'
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
    df_precip_scaled = data_precip[area_ratio_precip['label'].to_list()]
    for col in df_precip_scaled.columns:
        df_precip_scaled.loc[:, col] = df_precip_scaled[col] * area_ratio_precip[
            area_ratio_precip['label'] == col
            ]['updated_area_ratio'].iloc[0]
    df_precip_scaled = df_precip_scaled.sum(axis=1).to_frame()
    df_precip_scaled = df_precip_scaled[
        (df_precip_scaled.index >= (action_period.iloc[action_num]['start'] - pd.Timedelta(buffer, 'h')))
        & (df_precip_scaled.index <= (action_period.iloc[action_num]['end'] + pd.Timedelta(buffer, 'h')))
        ]

    # vis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df_dict[list(df_dict.keys())[0]].index,
            y=df_dict[list(df_dict.keys())[0]]['modeled'],
            mode='lines', name='Reported',
            line=dict(dash='solid', color='black')
        ),
        secondary_y=False,
    )
    for k, v in df_dict.items():
        fig.add_trace(
            go.Scatter(
                x=v.index, y=v['pred'].to_list(), mode='lines',
                name=models_label[k], line=dict(dash=models_symbol[k], color=models_color[k]),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Bar(
            x=df_precip_scaled.index, y=-df_precip_scaled[0],
            name="Rainfall", marker=dict(color='#808080')
        ),
        secondary_y=True,
    )
    max_discharge = round(df_dict[list(df_dict.keys())[0]]['modeled'].max())
    fig.update_yaxes(
        title_text="Streamflow (feet\u00B3)",
        showline=True,
        linewidth=2, linecolor='black', secondary_y=False,
        range=[0, max_discharge * 1.25],
        dtick=round((max_discharge * 2 / 6) / 1000) * 1000,
        tickformat=',',
        ticks='outside',
    )
    max_precip = round(df_precip_scaled[0].max())
    fig.update_yaxes(
        title_text="Rainfall (mm/h)", secondary_y=True,
        tickvals=[-i for i in list(range(0, 2 * max_precip + 1, round((max_precip * 2 / 5) / 10) * 10))],
        ticktext=[str(i) for i in list(range(0, 2 * max_precip + 1,  round((max_precip * 2 / 5) / 10) * 10))],
        range=[-max_precip * 2, 0],
        linewidth=2, linecolor='black',
        ticks='outside',
    )
    fig.update_layout(
        template='seaborn',
        xaxis_title='Time (EST)',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            dtick ='86400000',
            range=[
                action_period.iloc[action_num]['start'] - pd.Timedelta(buffer, 'h'),
                action_period.iloc[action_num]['end'] + pd.Timedelta(buffer, 'h')
            ]
        ),
        width=450, height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save_fig}/line_reported_vs_pred_{horizon}_action{action_num}.html')
    io.write_image(fig, f'{dir_save_fig}/line_reported_vs_pred_{horizon}_action{action_num}.png', scale=4)

if analysis_name == 'line_metric_field_modeled':
    dir_save_fig = './papers/figs'
    dir_data = './papers/tables'
    # df_overall_rc = pd.read_csv(f'{dir_data}/table_metric_overall_modeled.csv', index_col=[0, 1])
    # df_overall_rc = df_overall_rc.loc[(slice(None), 'pi_hodcrnn'), :]
    df_overall_rc = pd.read_csv(f'{dir_data}/table_metric_overall_modeled_selective.csv', index_col=[2])
    df_overall_rc = df_overall_rc.loc[('rc'), :]
    # df_high_rc = pd.read_csv(f'{dir_data}/table_metric_high_modeled.csv', index_col=[0, 1])
    # df_high_rc = df_high_rc.loc[(slice(None), 'pi_hodcrnn'), :]
    df_high_rc = pd.read_csv(f'{dir_data}/table_metric_high_modeled_selective.csv', index_col=[2])
    df_high_rc = df_high_rc.loc[('rc'), :]
    df_overall_field = pd.read_csv(f'{dir_data}/table_metric_overall_field.csv', index_col=[0, 1])
    df_high_field = pd.read_csv(f'{dir_data}/table_metric_high_field.csv', index_col=[0, 1])

    metric = [
        # 'MAE',
        # 'MAPE',
        # 'BIAS',
        'NSE'
    ]
    unit = [
        # '(feet\u00B3/s)',
        # '(%)',
        # '(feet\u00B3/s)',
        ''
    ]
    color_list = ['#386084', '#BF4163', '#992f87', '#efae42']
    horizon_list = [i + 1 for i in list(range(6))]

    overall_rc = df_overall_rc[metric[0]]
    overall_field = df_overall_field[metric[0]]
    high_rc = df_high_rc[metric[0]]
    high_field = df_high_field[metric[0]]

    # y_min = overall_rc.min() * 0.9
    # y_max = high_field.max() * 1.1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=horizon_list,
            y=overall_rc,
            mode='markers+lines',
            name='w.r.t. reported streamflow',
            line=dict(color=color_list[0]),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_list,
            y=overall_field,
            mode='markers+lines',
            name='w.r.t. measured streamflow',
            line=dict(color=color_list[1], dash='dash'),
        ),
    )
    fig.update_layout(
        template='seaborn',
        xaxis_title='Horizon',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title=f'Forecasting horizon (h)',
            dtick=1,
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title=f'{metric[0]} {unit[0]}',
            # range=[y_min, y_max],
        ),
        width=300, height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save_fig}/line_baseTune_{metric[0]}_vs_horizon_overall.html')
    io.write_image(fig, f'{dir_save_fig}/line_baseTune_{metric[0]}_vs_horizon_overall.png', scale=4)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=horizon_list,
            y=high_rc,
            mode='markers+lines',
            name='w.r.t. reported streamflow',
            line=dict(color=color_list[0]),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_list,
            y=high_field,
            mode='markers+lines',
            name='w.r.t. measured streamflow',
            line=dict(color=color_list[1], dash='dash'),
        ),
    )
    fig.update_layout(
        template='seaborn',
        xaxis_title='Horizon',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title=f'Forecasting horizon (h)',
            dtick=1,
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title=f'{metric[0]} {unit[0]}',
            # range=[y_min, y_max],
        ),
        width=300, height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save_fig}/line_baseTune_{metric[0]}_vs_horizon_high.html')
    io.write_image(fig, f'{dir_save_fig}/line_baseTune_{metric[0]}_vs_horizon_high.png', scale=4)

if analysis_name == 'scatter_original_error':
    import math
    dir_expr = './outputs/experiments/pi_hodcrnn_tune_o1_3__2024-04-28-14-12-57'
    dir_save = './papers/figs'
    target_gage = '01573560'
    horizon = 3

    train_val_df = pd.read_csv(f'{dir_expr}/train_val_df.csv')
    train_val_df['date_time'] = pd.to_datetime(train_val_df['date_time'], utc=True)
    train_val_df = train_val_df.set_index('date_time')
    train_val_df.index = train_val_df.index.tz_convert('America/New_York')
    train_val_df['dis_diff'] = train_val_df['pred_discharge'] - train_val_df['discharge']
    train_val_df['dis_diff_per'] = train_val_df['dis_diff'] / train_val_df['discharge']

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index =  pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['dis_diff'] = test_df['pred_w_o_tune'] - test_df['field']
    test_df['dis_diff_per'] = test_df['dis_diff'] / test_df['field']

    max_level = math.floor(max(train_val_df['pred_level'].max(), test_df['pred_water_level'].max()))
    max_error = math.ceil(max(train_val_df['dis_diff_per'].max(), test_df['dis_diff_per'].max()) * 100)
    min_error = math.floor(min(train_val_df['dis_diff_per'].min(), test_df['dis_diff_per'].min()) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df['pred_level'],
        y=train_val_df['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df['pred_water_level'],
        y=test_df['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_error_per_vs_level_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_error_per_vs_level_{horizon}_{target_gage}.png', scale=4)


    data_flood_stage = pd.read_csv(f'./data/USGS_gage_flood_stage/flood_stages.csv', dtype={'site_no': 'str'})
    data_flood_stage = data_flood_stage[data_flood_stage['site_no'] == target_gage]
    high_flow = data_flood_stage.iloc[0]['action']

    train_val_high_flow_index = (train_val_df[['last_past_value', 'pred_level']].values >= high_flow).any(axis=1)
    train_val_df.loc[:, 'per_level_change'] = (
            (train_val_df['pred_level'] - train_val_df['last_past_value']) / train_val_df['last_past_value']
    )
    test_df.loc[:, 'per_level_change'] = (
            (test_df['pred_water_level'] - test_df['last_past_value']) / test_df['last_past_value']
    )

    error_rate_array = np.concatenate((
            (train_val_df['pred_level'] - train_val_df['last_past_value']) / train_val_df['last_past_value'],
    ), axis=0)
    emergency_ratio = (train_val_high_flow_index.sum() / train_val_high_flow_index.shape[0])
    high_change = np.percentile(error_rate_array, (1-emergency_ratio) * 100)

    train_val_df_s_1 = train_val_df[(train_val_df[['pred_level', 'last_past_value']] >= high_flow).any(axis=1)]
    test_df_s_1 = test_df[(test_df[['pred_water_level', 'last_past_value']] >= high_flow).any(axis=1)]
    train_val_df_s_2 = train_val_df[
        train_val_df['per_level_change'].abs() >= high_change
    ].copy()
    train_val_df_s_2.loc[:, 'aux'] = [0] * len(train_val_df_s_2)
    test_df_s_2 = test_df[
        test_df['per_level_change'].abs() >= high_change
    ].copy()
    test_df_s_2.loc[:, 'aux'] = [0] * len(test_df_s_2)
    train_val_df_s = pd.merge(
        train_val_df_s_1, train_val_df_s_2[['aux']],
        left_index=True, right_index=True, how='inner'
    )
    test_df_s = pd.merge(
        test_df_s_1, test_df_s_2[['aux']],
        left_index=True, right_index=True, how='inner'
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df_s['pred_level'],
        y=train_val_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df_s['pred_water_level'],
        y=test_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_error_per_vs_level_select_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_error_per_vs_level_select_{horizon}_{target_gage}.png', scale=4)


    min_per_level_change = math.floor(
        min(train_val_df_s['per_level_change'].min(), test_df_s['per_level_change'].min()) * 100
    )
    max_per_level_change = math.ceil(
        max(train_val_df_s['per_level_change'].max(), test_df_s['per_level_change'].max()) * 100
    )

    x_range_list = list(range(min_per_level_change - 1, max_per_level_change + 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range_list,
        y=[0] * (max_per_level_change - min_per_level_change + 3),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df_s['per_level_change'] * 100,
        y=train_val_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df_s['per_level_change'] * 100,
        y=test_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Percentage gauge height change (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_per_level_change - 1, max_per_level_change + 1],
            dtick=(x_range_list[-1] - x_range_list[0]) // 7,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=450, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_error_per_vs_change_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_error_per_vs_change_{horizon}_{target_gage}.png', scale=4)

if analysis_name == 'scatter_line_tuner_o1':
    import math
    import pickle
    dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o1_1__2024-04-28-13-17-47'
    dir_save = './papers/figs'
    horizon = 1
    target_gage = '01573560'

    with open(f'{dir_expr}/tuner_o1_1degree_poly.pkl', 'rb') as file:
        model = pickle.load(file)

    # data
    train_val_df = pd.read_csv(f'{dir_expr}/train_val_df.csv')
    train_val_df['date_time'] = pd.to_datetime(train_val_df['date_time'], utc=True)
    train_val_df = train_val_df.set_index('date_time')
    train_val_df.index = train_val_df.index.tz_convert('America/New_York')
    train_val_df['dis_diff'] = train_val_df['pred_discharge'] - train_val_df['discharge']
    train_val_df['dis_diff_per'] = train_val_df['dis_diff'] / train_val_df['discharge']

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index =  pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['dis_diff'] = test_df['pred_w_o_tune'] - test_df['field']
    test_df['dis_diff_per'] = test_df['dis_diff'] / test_df['field']

    max_level = math.floor(max(train_val_df['pred_level'].max(), test_df['pred_water_level'].max()))
    max_error = math.ceil(max(train_val_df['dis_diff_per'].max(), test_df['dis_diff_per'].max()) * 100)
    min_error = math.floor(min(train_val_df['dis_diff_per'].min(), test_df['dis_diff_per'].min()) * 100)

    # level change
    # high_flow = (train_val_df['pred_level'].values.mean() + train_val_df['pred_level'].values.std() * 0.7)  # 6.11
    data_flood_stage = pd.read_csv(f'./data/USGS_gage_flood_stage/flood_stages.csv', dtype={'site_no': 'str'})
    data_flood_stage = data_flood_stage[data_flood_stage['site_no'] == target_gage]
    high_flow = data_flood_stage.iloc[0]['action']

    train_val_df.loc[:, 'per_level_change'] = (
            (train_val_df['pred_level'] - train_val_df['last_past_value']) / train_val_df['last_past_value']
    )
    test_df.loc[:, 'per_level_change'] = (
            (test_df['pred_water_level'] - test_df['last_past_value']) / test_df['last_past_value']
    )

    # high_change = (
    #         train_val_df['per_level_change'].values.mean()
    #         + train_val_df['per_level_change'].values.std() * 0.6)  # 0.015
    train_val_high_flow_index = (train_val_df[['last_past_value', 'pred_level']].values >= high_flow).any(axis=1)
    train_val_df.loc[:, 'per_level_change'] = (
            (train_val_df['pred_level'] - train_val_df['last_past_value']) / train_val_df['last_past_value']
    )
    test_df.loc[:, 'per_level_change'] = (
            (test_df['pred_water_level'] - test_df['last_past_value']) / test_df['last_past_value']
    )
    error_rate_array = np.concatenate((
            (train_val_df['pred_level'] - train_val_df['last_past_value']) / train_val_df['last_past_value'],
    ), axis=0)
    emergency_ratio = (train_val_high_flow_index.sum() / train_val_high_flow_index.shape[0])
    high_change = np.percentile(error_rate_array, (1-emergency_ratio) * 100)

    train_val_df_s_1 = train_val_df[(train_val_df[['pred_level', 'last_past_value']] >= high_flow).any(axis=1)]
    test_df_s_1 = test_df[(test_df[['pred_water_level', 'last_past_value']] >= high_flow).any(axis=1)]
    train_val_df_s_2 = train_val_df[
        train_val_df['per_level_change'].abs() >= high_change
    ].copy()
    train_val_df_s_2.loc[:, 'aux'] = [0] * len(train_val_df_s_2)
    test_df_s_2 = test_df[
        test_df['per_level_change'].abs() >= high_change
    ].copy()
    test_df_s_2.loc[:, 'aux'] = [0] * len(test_df_s_2)
    train_val_df_s = pd.merge(
        train_val_df_s_1, train_val_df_s_2[['aux']],
        left_index=True, right_index=True, how='inner'
    )
    test_df_s = pd.merge(
        test_df_s_1, test_df_s_2[['aux']],
        left_index=True, right_index=True, how='inner'
    )

    # range
    min_per_level_change = math.floor(
        min(train_val_df_s['per_level_change'].min(), test_df_s['per_level_change'].min()) * 100
    )
    max_per_level_change = math.ceil(
        max(train_val_df_s['per_level_change'].max(), test_df_s['per_level_change'].max()) * 100
    )

    # model
    model_x = np.array(list(range(min_per_level_change - 1, max_per_level_change + 2)))[:, np.newaxis]
    model_x = model_x / 100
    model_y = model.predict(model_x)

    x_range_list = list(range(min_per_level_change - 1, max_per_level_change + 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range_list,
        y=[0] * (max_per_level_change - min_per_level_change + 3),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=model_x[:, 0] * 100,
        y=model_y[:, 0] * 100,
        mode='lines',
        line=dict(width=3, dash='dash', color='#efae42'),
        name='Residual error learner'
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df_s['per_level_change'] * 100,
        y=train_val_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df_s['per_level_change'] * 100,
        y=test_df_s['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Percentage gauge height change (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_per_level_change - 1, max_per_level_change + 1],
            dtick=round((x_range_list[-1] - x_range_list[0]) / 7),
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=450, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_line_model_o1_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_line_model_o1_{horizon}_{target_gage}.png', scale=4)

if analysis_name == 'scatter_line_tuner_o2':
    import math
    import pickle
    dir_expr = './outputs/experiments/pi_hodcrnn_tune_o2_6__2024-04-28-18-23-15'
    dir_save = './papers/figs'
    target_gage = '01573560'
    horizon = 6

    with open(f'{dir_expr}/tuner_o2_lowess.pkl', 'rb') as file:
        model = pickle.load(file)

    # data
    train_val_df = pd.read_csv(f'{dir_expr}/train_val_df.csv')
    train_val_df['date_time'] = pd.to_datetime(train_val_df['date_time'], utc=True)
    train_val_df = train_val_df.set_index('date_time')
    train_val_df.index = train_val_df.index.tz_convert('America/New_York')
    train_val_df['dis_diff'] = train_val_df['pred_discharge'] - train_val_df['discharge']
    train_val_df['dis_diff_per'] = train_val_df['dis_diff'] / train_val_df['discharge']

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index =  pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['dis_diff'] = test_df['pred_w_o_tune'] - test_df['field']
    test_df['dis_diff_per'] = test_df['dis_diff'] / test_df['field']

    max_level = math.floor(max(train_val_df['pred_level'].max(), test_df['pred_water_level'].max()))
    max_error = math.ceil(max(train_val_df['dis_diff_per'].max(), test_df['dis_diff_per'].max()) * 100)
    min_error = math.floor(min(train_val_df['dis_diff_per'].min(), test_df['dis_diff_per'].min()) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df['pred_level'],
        y=train_val_df['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df['pred_water_level'],
        y=test_df['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_model_o1_result_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_model_o1_result_{horizon}_{target_gage}.png', scale=4)


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df['pred_level'],
        y=train_val_df['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df['pred_water_level'],
        y=test_df['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.add_trace(go.Scatter(
        x=model[:, 0],
        y=model[:, 1] * 100,
        mode='lines',
        line=dict(color='#efae42', width=3, dash='dash'),
        showlegend=False
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[min_error - 2, max_error + 2],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_model_o2_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_model_o2_{horizon}_{target_gage}.png', scale=4)

if analysis_name == 'scatter_post_o2_error':
    import math
    import pickle
    dir_expr = './outputs/experiments/pi_hodcrnn_tune_o3_3__2024-04-28-19-52-29'
    dir_save = './papers/figs'
    target_gage = '01573560'
    horizon = 3

    # data
    train_val_df = pd.read_csv(f'{dir_expr}/train_val_df.csv')
    train_val_df['date_time'] = pd.to_datetime(train_val_df['date_time'], utc=True)
    train_val_df = train_val_df.set_index('date_time')
    train_val_df.index = train_val_df.index.tz_convert('America/New_York')
    train_val_df['dis_diff'] = train_val_df['pred_discharge_o2'] - train_val_df['discharge']
    train_val_df['dis_diff_per'] = train_val_df['dis_diff'] / train_val_df['discharge']

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index =  pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['dis_diff'] = test_df['pred_w_o_tune'] - test_df['field']
    test_df['dis_diff_per'] = test_df['dis_diff'] / test_df['field']

    max_level = math.floor(max(train_val_df['pred_level'].max(), test_df['pred_water_level'].max()))
    max_error = math.ceil(max(train_val_df['dis_diff_per'].max(), test_df['dis_diff_per'].max()) * 100)
    min_error = math.floor(min(train_val_df['dis_diff_per'].min(), test_df['dis_diff_per'].min()) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df['pred_level'],
        y=train_val_df['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df['pred_water_level'],
        y=test_df['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            # range=[min_error - 2, max_error + 2],
            range=[-10, 15],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_model_o2_result_{horizon}_{target_gage}.html')
    io.write_image(fig, f'{dir_save}/scatter_model_o2_result_{horizon}_{target_gage}.png', scale=4)

if analysis_name == 'scatter_post_o3_error':
    import math
    import pickle
    dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_tune_o3_1__2024-03-16-17-43-59'
    dir_save = './papers/figs'
    horizon = 1

    # data
    train_val_df = pd.read_csv(f'{dir_expr}/train_val_df.csv')
    train_val_df['date_time'] = pd.to_datetime(train_val_df['date_time'], utc=True)
    train_val_df = train_val_df.set_index('date_time')
    train_val_df.index = train_val_df.index.tz_convert('America/New_York')
    train_val_df['dis_diff'] = train_val_df['pred_discharge_o3'] - train_val_df['discharge']
    train_val_df['dis_diff_per'] = train_val_df['dis_diff'] / train_val_df['discharge']

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index =  pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['dis_diff'] = test_df['pred'] - test_df['field']
    test_df['dis_diff_per'] = test_df['dis_diff'] / test_df['field']

    max_level = math.floor(max(train_val_df['pred_level'].max(), test_df['pred_water_level'].max()))
    max_error = math.ceil(max(train_val_df['dis_diff_per'].max(), test_df['dis_diff_per'].max()) * 100)
    min_error = math.floor(min(train_val_df['dis_diff_per'].min(), test_df['dis_diff_per'].min()) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(max_level + 2)), y=[0] * (max_level + 2),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_val_df['pred_level'],
        y=train_val_df['dis_diff_per'] * 100,
        mode='markers',
        name='Training set',
        marker=dict(color='#551C6E'),
    ))
    fig.add_trace(go.Scatter(
        x=test_df['pred_water_level'],
        y=test_df['dis_diff_per'] * 100,
        mode='markers',
        name='Test set',
        marker=dict(color='#93B237'),
    ))
    fig.update_layout(
        template='seaborn',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            title='Gauge height (ft)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            range=[0, max_level + 1],
            dtick=2,
        ),
        yaxis=dict(
            title='Percentage error (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            # range=[min_error - 2, max_error + 2],
            range=[-10, 15],
        ),
        width=550, height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    fig.write_html(f'{dir_save}/scatter_model_o3_result_{horizon}.html')
    io.write_image(fig, f'{dir_save}/scatter_model_o3_result_{horizon}.png', scale=4)

if analysis_name == 'table_metrics':
    import re
    dir_expr = './outputs/experiments'
    dir_save = './papers/tables'
    table_name = 'table_metric_overall_field_tuning'
    use_col = ['field_overall']
    model_list = [
    #     'naive', 'linear', 'mlp', 'xgboost', 'gru', 'hodcrnn',
        'pi_hodcrnn',
        'pi_hodcrnn_tune_o1', 'pi_hodcrnn_tune_o2', 'pi_hodcrnn_tune_o3',
        # 'pi_hodcrnn_tune_l',
    ]
    horizon_list = [
        1,
        2, 3, 4, 5, 6
    ]
    metric_list = [
        'MAE', 'MAPE', 'RMSE', 'BIAS', 'NSE',
        # 'CC',
        # 'PEAK_BIAS', 'PEAK_PER_BIAS', 'PEAK_TIME_BIAS'
    ]

    expr_list = [d for d in os.listdir(dir_expr) if os.path.isdir(os.path.join(dir_expr, d))
                       and d.startswith('ARCHIVE')]
    metric_list_100 = [
        'MAPE',
        # 'PEAK_PER_BIAS'
    ]
    metric_list_2digits = [
        'MAE', 'MAPE', 'RMSE', 'BIAS',
        # 'PEAK_BIAS', 'PEAK_PER_BIAS', 'PEAK_TIME_BIAS'
    ]
    metric_list_4digits = ['NSE', 'CC']
    metric_list_minus = ['PEAK_TIME_BIAS']
    expr_short_list = [re.search(rf"ARCHIVE_(.*?)__2024", e).group(1) for e in expr_list]
    df_dict = {}
    for h in horizon_list:
        row_list = []
        for m in model_list:
            folder_name = [e for e in expr_short_list if (e[:-len(e.split('_')[-1]) - 1] == m) and (e.split('_')[-1]) == str(h)]
            assert len(folder_name) == 1, 'Surplus experiments.'
            folder_name = expr_list[expr_short_list.index(folder_name[0])]

            df = pd.read_csv(f'{dir_expr}/{folder_name}/report_df.csv', index_col=0)
            df.loc[metric_list_100, :] = df.loc[metric_list_100, :] * 100
            df.loc[metric_list_2digits] = df.loc[metric_list_2digits, :].round(2)
            df.loc[metric_list_4digits] = df.loc[metric_list_4digits, :].round(4)
            if set(metric_list_minus).issubset(set(df.index)):
                df.loc[metric_list_minus] = -df.loc[metric_list_minus]
            row_df = df[use_col].T
            row_df.index = [m]
            row_list.append(row_df)
        df_cat = pd.concat(row_list, axis=0)
        df_cat = df_cat[metric_list]
        df_dict[h] = df_cat
    df_cat_cat = pd.concat(df_dict)
    df_cat_cat.to_csv(f'{dir_save}/{table_name}.csv')

if analysis_name == 'table_base_model_field_meas_time_points':

    import re
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_squared_error
    from utils.eval import metric_bias
    from utils.eval import metric_nse
    dir_expr = './outputs/experiments'
    dir_save = './papers/tables'
    table_name = 'table_metric_overall_modeled_selective'
    model_name = 'pi_hodcrnn_tune_base'
    horizon_list = [1, 2, 3, 4, 5, 6]
    metric_list = ['MAE', 'MAPE', 'RMSE', 'BIAS', 'NSE', ]
    metric_list_2digits = ['MAE', 'MAPE', 'RMSE', 'BIAS',]
    metric_list_4digits = ['NSE',]
    metric_list_100 = ['MAPE',]
    threshold = 0

    expr_list = [d for d in os.listdir(dir_expr) if os.path.isdir(os.path.join(dir_expr, d))
                 and d.startswith('ARCHIVE')]
    expr_short_list = [re.search(rf"ARCHIVE_(.*?)__2024", e).group(1) for e in expr_list]
    df_dict = {}
    row_list = []
    for h in horizon_list:
        folder_name = [e for e in expr_short_list if (e[:-len(e.split('_')[-1]) - 1] == model_name) and (e.split('_')[-1]) == str(h)]
        assert len(folder_name) == 1, 'Surplus experiments.'
        folder_name = expr_list[expr_short_list.index(folder_name[0])]

        df = pd.read_csv(f'{dir_expr}/{folder_name}/test_df.csv', index_col=0)

        modeled = df['modeled'][df['water_level'] >= threshold]
        pred = df['pred'][df['water_level'] >= threshold]
        field = df['field'][df['water_level'] >= threshold]

        metric_rc = [
            mean_absolute_error(modeled, pred),
            mean_absolute_percentage_error(modeled, pred),
            mean_squared_error(modeled, pred, squared=False),
            metric_bias(modeled, pred),
            metric_nse(modeled, pred),
        ]
        row_df_rc = pd.DataFrame([[h, 'rc'] + metric_rc], columns=['horizon', 'target']+metric_list)
        metric_field = [
            mean_absolute_error(field, pred),
            mean_absolute_percentage_error(field, pred),
            mean_squared_error(field, pred, squared=False),
            metric_bias(field, pred),
            metric_nse(field, pred),
        ]
        row_df_field = pd.DataFrame([[h, 'field'] + metric_field], columns=['horizon', 'target']+metric_list)
        row_list.append(row_df_rc)
        row_list.append(row_df_field)

    df_cat = pd.concat(row_list, axis=0)
    df_cat.loc[:, metric_list_100] = df_cat.loc[:, metric_list_100] * 100
    df_cat.loc[:, metric_list_2digits] = df_cat.loc[:, metric_list_2digits].round(2)
    df_cat.loc[:, metric_list_4digits] = df_cat.loc[:, metric_list_4digits].round(4)
    df_cat.to_csv(f'{dir_save}/{table_name}.csv')


if analysis_name == 'dist_accuracy':
    import plotly.figure_factory as ff
    horizon = 1
    dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_1__2024-03-14-17-37-25'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_2__2024-03-14-17-34-33'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_3__2024-03-14-17-32-00'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_4__2024-03-14-17-29-51'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_5__2024-03-14-17-24-40'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_6__2024-03-14-17-22-44'
    dir_save = './papers/figs'

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index = pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['w_field'] = (test_df['pred'] - test_df['field'])
    test_df['w_modeled'] = (test_df['pred'] - test_df['modeled'])

    test_df_full = pd.read_csv(f'{dir_expr}/test_df_full.csv', index_col=0)
    test_df_full.index = pd.to_datetime(test_df_full.index, utc=True).tz_convert('America/New_York')
    test_df_full['w_modeled'] = (test_df_full['pred'] - test_df_full['modeled'])

    labels = ['w.r.t. Measurement', 'w.r.t. RC generation',]

    fig = ff.create_distplot(
        [
            test_df['w_field'].to_list(),
            # test_df_full['w_modeled'][~test_df_full['w_modeled'].isna()]
            test_df['w_modeled'].to_list(),
        ],
        labels,
        bin_size=2,
        curve_type='kde',
        show_hist=False,
        show_rug=False,
        colors=['#992f87', '#efae42'],
    )
    fig.update_layout(
        template='seaborn',
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title='Error',
            range=[-1000, 1000]
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            ticks='outside',
            title='Density',
            mirror='all',
            side='left',
            range=[0, 0.01]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=300, height=400,
    )
    fig.write_html(f'{dir_save}/dist_pred_modeled_field_{horizon}.html')
    io.write_image(fig, f'{dir_save}/dist_pred_modeled_field_{horizon}.png', scale=4)

if analysis_name == 'dist_accuracy_bk':

    horizon = 4
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_1__2024-03-14-17-37-25
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_2__2024-03-14-17-34-33'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_3__2024-03-14-17-32-00'
    dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_4__2024-03-14-17-29-51'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_5__2024-03-14-17-24-40'
    # dir_expr = './outputs/experiments/ARCHIVE_pi_hodcrnn_6__2024-03-14-17-22-44'
    dir_save = './papers/figs'

    test_df = pd.read_csv(f'{dir_expr}/test_df.csv', index_col=0)
    test_df.index = pd.to_datetime(test_df.index, utc=True).tz_convert('America/New_York')
    test_df['w_modeled'] = (test_df['pred'] - test_df['modeled']) / test_df['modeled']
    test_df['w_field'] = (test_df['pred'] - test_df['field']) / test_df['field']

    labels = ['w.r.t. RC generation', 'w.r.t. Measurement',]
    colors = ['#992f87', '#efae42']

    fig = go.Figure()
    fig.add_trace(go.Violin(
        x=[3] * len(test_df),
        y=test_df['w_modeled'],
        legendgroup='Yes',
        scalegroup='Yes',
        name=labels[0],
        side='negative',
        line_color=colors[0],

    ))
    fig.add_trace(go.Violin(
        x=[3] * len(test_df),
        y=test_df['w_field'],
        legendgroup='Yes',
        scalegroup='Yes',
        name=labels[1],
        side='positive',
        line_color=colors[1],
    ))
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay')

    fig.update_layout(
        template='seaborn',
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            title='Absolute error',
            # range=[-0.1, 0.1],
            # range=[0, 1]
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            ticks='outside',
            title='Density',
            mirror='all',
            side='left'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=400, height=500,
    )
    fig.write_html(f'{dir_save}/dist_pred_modeled_field_{horizon}.html')
    io.write_image(fig, f'{dir_save}/dist_pred_modeled_field_{horizon}.png', scale=4)


if analysis_name == 'box_pred_error':
    saved_dir = './outputs/figs/'
    dir_expr = [
        './outputs/experiments/ARCHIVE_naive_2023-11-28-23-52-15',
        # './outputs/experiments/ARCHIVE_arima_2023-11-29-01-18-31',
        './outputs/experiments/ARCHIVE_linear_2023-11-29-20-20-48',
        './outputs/experiments/ARCHIVE_lstm_2023-11-30-15-56-28',
        './outputs/experiments/ARCHIVE_lstm_tune_2023-11-29-23-41-35',
        './outputs/experiments/ARCHIVE_pilstm_tune_2023-11-29-20-43-26',
    ]

    # data
    df = pd.read_csv(dir_expr[0] + '/test_df.csv').rename(columns={'Unnamed: 0': 'datetime'}).drop('pred', axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    df_vertical_list = []
    for expr in dir_expr:
        model_name = expr.split('ARCHIVE_')[1].split('_2')[0]
        test_df = pd.read_csv(expr + '/test_df.csv').rename(columns={'Unnamed: 0': 'datetime'})
        test_df['type'] = [model_name] * len(test_df)
        test_df['datetime'] = pd.to_datetime(test_df['datetime'])
        test_df = test_df.set_index('datetime')

        if 'pred' not in test_df.columns:
            test_df = test_df.drop('pred_w_o_tune', axis=1).rename(columns={'pred_tune': 'pred'})

        df[model_name] = test_df['pred']

        df_vertical_list.append(test_df)
    df_vertical = pd.concat(df_vertical_list, axis=0)
    df_vertical['perc_error'] = (df_vertical['pred'] - df_vertical['field']).abs() / df_vertical['field']

    # rename
    df_vertical = df_vertical.replace('naive', 'Persistence Model')
    df_vertical = df_vertical.replace('linear', 'Linear Regression')
    df_vertical = df_vertical.replace('lstm', 'LSTM')
    df_vertical = df_vertical.replace('lstm_tune', 'Fine-tuned LSTM')
    df_vertical = df_vertical.replace('pilstm_tune', 'Fine-tuned PILSTM')

    # plot
    fig = px.box(df_vertical, y="type", x="perc_error", color="type",
                 points='all', template='seaborn')
    fig.update_traces(quartilemethod="inclusive")
    fig.update_layout(yaxis_title='Percentage Error',
                      xaxis_title='Model',
                      showlegend=False,
                      )
    fig.write_html(saved_dir + 'box_pred_error.html')

if analysis_name == 'model_explain':
    expr = './outputs/experiments/pilstm_tune_2023-11-30-22-25-19'
    saved_dir = './outputs/figs/'
    model = torch.load(expr + '/model.pth')

    # resume data
    test_df_full = pd.read_csv(expr + '/test_df_full.csv')
    test_df = test_df_full[~test_df_full['field'].isna()]
    test_x = test_df[[col for col in test_df.columns if 'lag' in col]].values[:, :, np.newaxis]
    test_x = torch.tensor(test_x).to(next(model.parameters()).device).float()

    # run model
    _, (hn, cn) = model.LSTM(test_x)
    hn_to_dense = torch.cat([hn[i] for i in range(hn.size(0))], dim=-1)
    wl = model.dense(hn_to_dense)
    pred_raw_dis = model.rc_direct(wl) * (model.norm_param[1] - model.norm_param[0]) + model.norm_param[0]
    pred_residual_rc = model.flex_rc(wl)
    pred_residual_unsteady = model.flex_unsteady(hn_to_dense)

    # organize data
    test_df['pred_raw_dis'] = pred_raw_dis.cpu().detach().numpy()
    test_df['pred_residual_rc'] = pred_residual_rc.cpu().detach().numpy()
    test_df['pred_residual_unsteady'] = pred_residual_unsteady.cpu().detach().numpy()
    test_df_short_1 = test_df.copy()[[col for col in test_df.columns if 'lag' not in col]]
    test_df_short_2 = test_df.copy()[['pred_residual_unsteady'] + [col for col in test_df.columns if 'lag' in col]]

    # unsteady flow
    test_df_short_3 = test_df.copy()[[col for col in test_df.columns if 'lag' in col]]
    past_dis = model.rc_direct(test_x[:, :, :]) * (model.norm_param[1] - model.norm_param[0]) + model.norm_param[0]
    past_dis = past_dis.cpu().detach().numpy()
    test_df_short_3.iloc[:, :] = past_dis[:, :, 0]

    test_df_short_3['field'] = test_df['field']
    test_df_short_3['modeled'] = test_df['modeled']
    test_df_short_3['unsteady_rate'] = (test_df_short_3['lag_0'] - test_df_short_3['lag_1']) / test_df_short_3['lag_1']
    # test_df_short_3['unsteady_rate_baseline'] = (test_df_short_3['field'] - test_df_short_3['modeled']) / test_df_short_3['field']
    # test_df_short_3['unsteady_rate_delta'] = (test_df_short_3['unsteady_rate'] - test_df_short_3['unsteady_rate_baseline'])
    test_df_short_3['pred_residual_unsteady'] = pred_residual_unsteady.cpu().detach().numpy()

    # test_df_short_3 = test_df_short_3[test_df_short_3['unsteady_rate'].abs() > 0.1]
    fig = px.scatter(x=test_df_short_3['unsteady_rate'] * 100, y=test_df_short_3['pred_residual_unsteady'],
                     trendline="ols")
    fig.update_layout(template='seaborn',
                      yaxis_title='Remedies by Fine Tuning (ft\u00B3)',
                      xaxis_title='Rate of Change of Discharge (%)',
                      showlegend=False,
                      )
    fig.write_html(saved_dir + 'scatter_unsteady_vs_residual.html')

    # modeled vs field
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(test_df_short_1))), y=test_df_short_1['field'].sort_values(),
                             mode='lines', name='Field Measured Discharge'))
    fig.add_trace(go.Scatter(x=list(range(len(test_df_short_1))),
                             y=(test_df_short_1['modeled'] + test_df_short_1['pred_residual_rc']).sort_values(),
                             mode='lines', name='Adjusted Discharge'))
    fig.add_trace(go.Scatter(x=list(range(len(test_df_short_1))), y=test_df_short_1['modeled'].sort_values(),
                             mode='lines', name='Modeled Discharge'))
    fig.update_layout(template='seaborn',
                      yaxis_title='Discharge (ft\u00B3)',
                      xaxis_title='Test Data Points Index',
                      )
    fig.write_html(saved_dir + 'line_modeled_field_rc.html')

print()
