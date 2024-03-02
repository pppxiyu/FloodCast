import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import utils.preprocess as pp

analysis_name = 'model_explain'
saved_dir = './outputs/figs/'

if analysis_name == 'box_pred_error':

    expr_dir = [
        './outputs/experiments/ARCHIVE_naive_2023-11-28-23-52-15',
        # './outputs/experiments/ARCHIVE_arima_2023-11-29-01-18-31',
        './outputs/experiments/ARCHIVE_linear_2023-11-29-20-20-48',
        './outputs/experiments/ARCHIVE_lstm_2023-11-30-15-56-28',
        './outputs/experiments/ARCHIVE_lstm_tune_2023-11-29-23-41-35',
        './outputs/experiments/ARCHIVE_pilstm_tune_2023-11-29-20-43-26',
    ]

    # data
    df = pd.read_csv(expr_dir[0] + '/test_df.csv').rename(columns={'Unnamed: 0': 'datetime'}).drop('pred', axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    df_vertical_list = []
    for expr in expr_dir:
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
