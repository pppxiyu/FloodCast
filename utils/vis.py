import plotly.graph_objects as go
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

import numpy as np


def plot_lines(df, select_col, save_dir, yaxis_title='Value'):
    # USE: takes in df and draw the selected cols in line chart
    # INPUT: df
    #        save_dir, str
    #        select_col, list of str
    # OUTPUT: html file

    df = df[select_col]

    fig = go.Figure()
    for col_name, data in df.items():
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines'))
    fig.update_layout(xaxis_title="Time", yaxis_title=yaxis_title)

    fig.write_html(save_dir)

    return


def plot_lines_surge(df, water_col, surge_col, save_dir, yaxis_title='Water level'):
    # USE: draw water level line and surge
    # INPUT: df
    #        save_dir, str
    #        water_col, str
    # OUTPUT: html file

    surge = df[df[surge_col] == 1].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[water_col].index, y=df[water_col].values, mode='lines'))
    fig.add_trace(go.Scatter(x=surge.index, y=surge[water_col].values, mode='markers'))

    fig.update_layout(xaxis_title="Time", yaxis_title=yaxis_title)
    fig.write_html(save_dir)

    return


def plot_bar_pac_matplot(series, title, if_ac=False):
    # USE: plot the ac of a df series
    # INPUT: df series
    # OUTPUT: show plot

    plt.figure()
    if not if_ac:
        plot_pacf(series.values, lags=100, method='ywm')
    else:
        plot_acf(series.values, lags=100, method='ywm')
    plt.title(title)
    plt.show()

    return


def plot_bar_pac(series, title, save_dir, if_ac=False):
    # USE: plot the ac of a df series
    # INPUT: df series
    # OUTPUT: save html plot

    if if_ac:
        cf = acf(series.values, nlags=120, alpha=0.05)
    else:
        cf = pacf(series.values, nlags=120, alpha=0.05)

    lower_b = cf[1][:, 0] - cf[0]
    upper_b = cf[1][:, 1] - cf[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x, x), y=(0, cf[0][x]), mode='lines', line_color='#3f3f3f') for x in range(len(cf[0]))]
    fig.add_scatter(x=np.arange(len(cf[0])), y=cf[0], mode='markers', marker_color='#1f77b4', marker_size=5)
    fig.add_scatter(x=np.arange(len(cf[0])), y=upper_b, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(cf[0])), y=lower_b, mode='lines', fillcolor='rgba(32,146,230,0.3)',
                    fill='tonexty',
                    line_color='rgba(255,255,255,0)')

    fig.update_traces(showlegend=False)
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(title=title)
    fig.update_layout(xaxis_title='Timestamp')
    fig.update_layout(width=600, height=666)
    fig.update_layout(font_size=14)
    fig.write_html(save_dir)

    return
