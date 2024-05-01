import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    from statsmodels.graphics.tsaplots import plot_pacf
    from statsmodels.graphics.tsaplots import plot_acf

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
    from statsmodels.tsa.stattools import pacf
    from statsmodels.tsa.stattools import acf

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

def plot_ridge_rc_error(df, save_dir):

    import plotly.figure_factory as ff
    import plotly.io as io

    labels = ['Normal', 'Action', 'Minor', 'Moderate', 'Major']
    over_5_per = [
        round((len(df[col][df[col] > 5].dropna()) / len(df[col].dropna())) * 100, 2)
        for col in df.columns
    ]
    over_5_per_height = [.02, .075, .075, .075, .075]
    over_5_per_y_offset = [.08, .13, .15, .17, .19]

    fig = ff.create_distplot(
        [df[col][df[col] <= 15].dropna().to_list() for col in df.columns],
        labels,
        bin_size=0.4,
        show_hist=False,
        show_rug=False,
        colors=['#34638B', '#fed98e', '#fe9929', '#d95f0e', '#993404'],
    )
    fig.add_shape(
        type="line",
        x0=5, y0=.02, x1=5, y1=.92,
        line=dict(color="Grey", width=1, dash="dash",),
        yref="paper",
    )
    fig.add_annotation(
        x=5, y=.98, yref="paper",
        text="5%",
        showarrow=False,
    )
    for p, l, h, o in zip(over_5_per, labels, over_5_per_height, over_5_per_y_offset):
        fig.add_annotation(
            x=5.5, y=h, xanchor="left",
            text=f"{l}: {p}%",
            showarrow=True,
            arrowhead=1, arrowcolor='Grey',
            ax=6, axref='x',
            ay=o, ayref='y',
        )
    fig.add_annotation(
        x=6, y=.22,
        text='Percentage over MAPE 5%: ',
        showarrow=False,
        xanchor="left",
    )
    fig.update_layout(
        template='seaborn',
        xaxis_title='MAPE (%)',
        yaxis_title='Density',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside', tickmode='linear',
            tickformat=',',
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror='all',
            range=[0, 0.4],dtick=0.05,
            ticks='outside', tickmode='linear',
            tickformat=',',
        ),
        width=600, height=450,
        legend=dict(x=0.75, y=0.99,),
    )
    fig.write_html(f'{save_dir}/dist_rc_error.html')
    io.write_image(fig, f'{save_dir}/dist_rc_error.png', scale=4)

    return


def plot_map_nodes(rivers, anchor, grid, anchor_adjusted, basin, nodes_on_rivers, dir_save, gauge_nodes=None):

    import shapely

    with open('./utils/mapbox_token.txt', "r") as f:
        mb_token = f.read().strip()
    pass

    # river lats/lons
    rivers_lats = []
    rivers_lons = []
    names = []
    for feature, name in zip(rivers.geometry, rivers.HYRIV_ID):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            rivers_lats = np.append(rivers_lats, y)
            rivers_lons = np.append(rivers_lons, x)
            names = np.append(names, [name] * len(y))
            rivers_lats = np.append(rivers_lats, None)
            rivers_lons = np.append(rivers_lons, None)
            names = np.append(names, None)

    fig = go.Figure()
    # basin
    fig.add_trace(go.Scattermapbox(
        fill="toself",
        lon=[point[0] for point in basin.geometry.values[0].exterior.coords],
        lat=[point[1] for point in basin.geometry.values[0].exterior.coords],
        marker=dict(size=2, color='#7393B3'),
        name="Watershed",
        showlegend=False,
    ))
    # rivers
    fig.add_trace(go.Scattermapbox(
        mode = "lines",
        lon=rivers_lons,
        lat=rivers_lats,
        marker=dict(size=4, color='#0F52BA'),
        name="Rivers",
        showlegend=False,
    ))
    # grid
    for cell in grid.cell:
        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon=[point[0] for point in cell.exterior.coords],
            lat=[point[1] for point in cell.exterior.coords],
            marker=dict(size=1, color='#6F8FAF'),
            name="Grid",
            showlegend=False,
        ))
    # anchor points
    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon=anchor.geometry.x.to_list(),
        lat=anchor.geometry.y.to_list(),
        marker=dict(size=5, color='#48BBDB'),
        name="Cell Centers",
    ))
    # adjusted points
    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon=anchor_adjusted.geometry.x.to_list(),
        lat=anchor_adjusted.geometry.y.to_list(),
        marker=dict(size=5, color='#006FB9'),
        name="Adjusted Cell Centers",
    ))
    # nodes on rivers
    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon=nodes_on_rivers.geometry.x.to_list(),
        lat=nodes_on_rivers.geometry.y.to_list(),
        marker=dict(size=9, color='#FFBF00'),
        name="Equivalent Graph Nodes",
    ))
    # gauging stations nodes
    if gauge_nodes is not None:
        fig.add_trace(
            go.Scattermapbox(
                mode="markers",
                lon=gauge_nodes.geometry.x.to_list(),
                lat=gauge_nodes.geometry.y.to_list(),
                marker=dict(size=7, color="#FF5733"),
                name="Gauging Stations",
            )
        )
    # Update the layout
    fig.update_layout(
        mapbox = {
            "style": "carto-positron",
            "center": {"lon": -76.45, "lat": 40.46},
            "zoom": 9,
            'accesstoken': mb_token},
        showlegend=True,
        legend = dict(
            x=0.75,
            y=0.95,
            xanchor="left",
            yanchor="top",
        )
    )
    fig.write_html(dir_save)

    return

