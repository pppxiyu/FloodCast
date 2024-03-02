import pandas as pd
import os
import torch

import utils.preprocess as pp
import utils.modeling as mo
import utils.vis as vis

import plotly.express as px
import plotly.graph_objects as go


analysis_name = 'batch_pull_up_gages'
dir_save_fig = './outputs/figs/'


if analysis_name == 'field_vs_model_all_flooding_rivers':

    dir_major_flood_riv = './outputs/USGS_gaga_filtering'
    dir_USGS_field = './data/USGS_gage_field'

    flood_major_riv_gage = pd.read_csv(dir_major_flood_riv + '/gauge_flood_counts.csv', dtype={'SITENO': 'str'})
    pp.pull_USGS_gage_field(dir_USGS_field, flood_major_riv_gage)

    if 'dis_modeled_error' in flood_major_riv_gage.columns:
        gage_list = flood_major_riv_gage[flood_major_riv_gage['dis_modeled_error'].isna()]['SITENO']
    else:
        gage_list = flood_major_riv_gage['SITENO']
    for gage in gage_list:
    # for gage in flood_major_riv_gage['SITENO']:
        try:
            data = pp.import_data_simplified(f'./data/USGS_gage_iv/{gage}.csv')
            data_field = pp.import_data_field(f'./data/USGS_gage_field/{gage}.csv')
            if data_field is None:
                # print(f"No field measurement for gage {gage}")
                continue
            data_field_modeled = pp.merge_field_modeled(data_field, data)
            if data_field_modeled is None:
                # print(f'Gauge {gage} does not have discharge iv data.')
                continue

            ave_error = data_field_modeled['perc_error'].abs().mean()
            flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'dis_modeled_error'] = ave_error

            if (data_field_modeled is not None and
                    ('water_level' in data_field_modeled.columns or
                     'water_level_adjusted' in data_field_modeled.columns)):
                water_level_col = [w for w in ['water_level', 'water_level_adjusted'] if
                                   w in data_field_modeled.columns]

                action_flood_stage = flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'action'].values[0]
                flood_flood_stage = flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'flood'].values[0]
                moderate_flood_stage = flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'moderate'].values[0]
                major_flood_stage = flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'major'].values[0]

                ave_error_action = data_field_modeled[data_field_modeled[water_level_col[0]] >= action_flood_stage][
                    'perc_error'].mean()
                ave_error_flood = data_field_modeled[data_field_modeled[water_level_col[0]] >= flood_flood_stage][
                    'perc_error'].mean()
                ave_error_moderate = data_field_modeled[data_field_modeled[water_level_col[0]] >= moderate_flood_stage][
                    'perc_error'].mean()
                ave_error_major = data_field_modeled[data_field_modeled[water_level_col[0]] >= major_flood_stage][
                    'perc_error'].mean()

                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'dis_modeled_error_action'] = ave_error_action
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'dis_modeled_error_moderate'] = ave_error_moderate
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'dis_modeled_error_flood'] = ave_error_flood
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'dis_modeled_error_major'] = ave_error_major

                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'records_count'] = len(
                    data_field_modeled)
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'records_count_action'] = len(
                    data_field_modeled[data_field_modeled[water_level_col[0]] >= action_flood_stage])
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'records_count_flood'] = len(
                    data_field_modeled[data_field_modeled[water_level_col[0]] >= flood_flood_stage])
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'records_count_moderate'] = len(
                    data_field_modeled[data_field_modeled[water_level_col[0]] >= moderate_flood_stage])
                flood_major_riv_gage.loc[flood_major_riv_gage['SITENO'] == gage, 'records_count_major'] = len(
                    data_field_modeled[data_field_modeled[water_level_col[0]] >= major_flood_stage])
        except:
            print(f'Analysis for gage {gage} failed.')

    flood_major_riv_gage.to_csv(dir_major_flood_riv + '/gauge_flood_counts.csv', index=False)

    pass

if analysis_name == 'field_vs_model_all_flooding_rivers_vis':
    dir_major_flood_riv = './outputs/USGS_gaga_filtering'
    save_dir = './papers/figs'

    rc_modeling_error = pd.read_csv(dir_major_flood_riv + '/gauge_flood_counts.csv', dtype={'SITENO': 'str'})
    rc_modeling_error = rc_modeling_error[~rc_modeling_error.duplicated('SITENO')]
    rc_modeling_error = rc_modeling_error[~rc_modeling_error['dis_modeled_error'].isna()]
    rc_modeling_error = rc_modeling_error[
        [col for col in rc_modeling_error.columns if 'dis_modeled_error' in col]
    ]
    vis.plot_ridge_rc_error(rc_modeling_error, save_dir)

    pass


if analysis_name == 'field_vs_modeled':
    num_error_rate_class = 7

    # import data
    data = pp.import_data('./data/USGS_gage_05311000/05311000_iv.csv')[['05311000_00065', '05311000_00060']]
    data.columns = ['water_level', 'discharge']
    data_field = pp.import_data_field('./data/USGS_gage_field/05311000.csv')

    data_field_modeled = pp.merge_field_modeled(data_field, data)
    data_field_modeled['perc_error_range'] = pd.qcut(data_field_modeled['perc_error'],
                                                     num_error_rate_class, precision=1)
    data_field_modeled = data_field_modeled.sort_values(by='perc_error_range',
                                                        key=lambda x: x.apply(lambda interval: interval.left))

    fig = px.scatter(data_field_modeled, y='discharge_modeled', x='discharge_field',
                     color='perc_error_range', color_discrete_sequence=px.colors.sequential.Blues)
    fig.add_trace(go.Scatter(x=data_field_modeled['discharge_modeled'], y=data_field_modeled['discharge_modeled'],
                             mode='lines', showlegend=False))
    fig.update_layout(template='seaborn',
                      yaxis_title='Modeled discharge (feet\u00B3)',
                      xaxis_title='Measured discharge (feet\u00B3)',
                      showlegend=True,
                      legend_title_text='Percentage Error (%)'
                      )
    fig.add_annotation(x=1500, y=1500, text="y=x", showarrow=True, arrowsize=1.5, arrowwidth=1.5)
    fig.add_annotation(xref="paper", x=1, xanchor='right', y=-50,
                       text=f"MAPE: {round(data_field_modeled['perc_error'].mean(), 2)}%", showarrow=False)
    fig.write_html(dir_save_fig + 'scatter_field_vs_modeled.html')

if analysis_name == 'pull_JAXA_data':

    import os
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    uid = 'rainmap'
    psd = 'Niskur+1404'

    dts = pd.date_range(
        start='1/1/2007',
        end='01/01/2024',
        freq='4MS',
        tz='America/New_York'
    ).tz_convert('UTC').strftime('%Y%m%d%H').to_list()
    st_list = dts[:-1]
    ed_list = dts[1:]

    lat_list = ['40.2', '40.3', '40.4', '40.5', '40.6']
    lon_list = ['-76.8', '-76.7', '-76.6', '-76.5', '-76.4', '-76.3']

    csv_files = [file for file in os.listdir('C:/Users/xpan88\Downloads') if file.endswith('.csv')]
    saved_file = [(
        i[i.find('_st') + 3: i.find('_ed')],
        i[i.find('_ed') + 3: i.find('_clat')],
        i[i.find('_clat') + 5: i.find('_clon')],
        i[i.find('_clon') + 5: i.find('.csv')],
    ) for i in csv_files]

    initial = False
    for st, ed in zip(st_list, ed_list):
        for lat in lat_list:
            for lon in lon_list:
                if (st, ed, lat, lon) not in saved_file:
                    # open webpage
                    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                    head = 'https://sharaku.eorc.jaxa.jp/cgi-bin/trmm/GSMaP/tilemap/show_graph.cgi?flag=1&'

                    url = f"{head}st={st}&ed={ed}&lat0={lat}&lon0={lon}&lang=en"
                    driver.get(url)
                    button = driver.find_element(By.ID, 'graph_dl')
                    button.click()

                    # open csv download window
                    original_window = driver.current_window_handle
                    assert len(driver.window_handles) > 1, "No new window opened"
                    new_window = [window for window in driver.window_handles if window != original_window][0]
                    driver.switch_to.window(new_window)

                    # input uid and psd
                    new_url = driver.current_url
                    update_new_url = new_url.split('//')[0] + f'//{uid}:{psd}@' + new_url.split('//')[1]
                    driver.get(update_new_url)

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

                    saved_file.append((st, ed, lat, lon))
                    initial = True
                    break
            if initial:
                break
        if initial:
            break
    if not initial:
        raise EOFError('All files have been pulled.')

    first_run = True
    for st, ed in zip(st_list, ed_list):
        for lat in lat_list:
            for lon in lon_list:
                if (st, ed, lat, lon) not in saved_file:
                    if first_run:
                        first_run = False
                        continue
                    url = f"{head}st={st}&ed={ed}&lat0={lat}&lon0={lon}&lang=en"
                    driver.get(url)
                    button = driver.find_element(By.ID, 'graph_dl')
                    button.click()

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

    time.sleep(10)
    driver.quit()
    pass

if analysis_name == 'organize_JAXA_data':

    import os
    import pandas as pd

    working_dir = './data/JAXA_precipitation_data'

    csv_files = [file for file in os.listdir(working_dir) if file.endswith('.csv')]
    saved_file = [(
        i[i.find('out') + 3: i.find('_st')],
        i[i.find('_st') + 3: i.find('_ed')],
        i[i.find('_ed') + 3: i.find('_clat')],
        i[i.find('_clat') + 5: i.find('_clon')],
        i[i.find('_clon') + 5: i.find('.csv')],
    ) for i in csv_files]
    loc_list = list(set([i[3:5] for i in saved_file]))

    concatenated_csv_files = [file for file in os.listdir(f'{working_dir}/concatenated') if file.endswith('.csv')]
    concatenated_saved_files = [(
        i[i.find('clat') + 4: i.find('_clon')],
        i[i.find('_clon') + 5: i.find('.csv')],
    ) for i in concatenated_csv_files]

    for loc in loc_list:
        if loc not in concatenated_saved_files:
            loc_files = [i for i in saved_file if i[3:5] == loc]
            f_list = []
            for loc_file in loc_files:
                f = pd.read_csv(
                    f'{working_dir}/out{loc_file[0]}_st{loc_file[1]}_ed{loc_file[2]}_clat{loc_file[3]}_clon{loc_file[4]}.csv',
                    usecols=['date', 'value'],
                )
                f_list.append(f)
            ff = pd.concat(f_list, axis=0)
            ff['date'] = pd.to_datetime(ff['date'], utc=True)
            ff = ff.drop_duplicates()
            ff = ff.set_index('date')
            ff = ff.asfreq('H')
            # check if being consecutive
            if ff.index.to_list() != pd.date_range(start=ff.index.min(), end=ff.index.max(), freq='H').to_list():
                print('Datetime index is not consecutive or consistent!')
                print()
            ff.to_csv(f'{working_dir}/concatenated/clat{loc[0]}_clon{loc[1]}.csv')

if analysis_name == 'line_rating_curve':
    df = data.resample('H', closed='right', label='right').mean()

    for filename in os.listdir('.'):
        if filename.startswith(f'saved_best_adapted_direct_') and filename.endswith('.pth'):
            rc_pretrain = torch.load(filename)

    x_rc, y_rc = mo.rc_run(rc_pretrain, x_min=df['water_level'].min(), x_max=df['water_level'].max(),
                           y_min=df['discharge'].min(), y_max=df['discharge'].max(), step=0.1)
    fig = px.line(x=x_rc, y=y_rc, labels={'x': 'Water Level (ft)', 'y': 'Discharge (ft\u00B3)'})
    fig.update_layout(template='seaborn', width=300, )
    fig.write_html(dir_save_fig + 'line_rating_curve.html')

if analysis_name == 'test_USGS_API':

    import requests
    import datetime
    import time
    import pandas as pd
    import math

    def get_water_data(site_number="02341460"):
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_number}&parameterCd=00065,00045,00060&siteStatus=all&period=P1D"
        response = requests.get(url)

        if response.status_code == 200:

            response_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = response.json()
            time_series_data = {}
            parameters = ['Gage height, ft', 'Precipitation, total, in', 'Streamflow, ft&#179;/s']

            for time_series in data['value']['timeSeries']:
                parameter = time_series['variable']['variableName']
                values = time_series['values'][0]['value']
                time_series_data[parameter] = {
                    point['dateTime'].replace(':00.000-04:00', '').replace('T', ' '): float(point['value']) for point in
                    values}

            combined_data = []
            timestamps = []

            for timestamp in sorted(time_series_data[parameters[0]]):
                row = [time_series_data[param].get(timestamp, None) for param in parameters]
                combined_data.append(row)
                timestamps.append(timestamp)

            return combined_data[-1], timestamps[-1], response_timestamp

        else:
            return None, None, None

    interval = 10

    time_data = []
    time_response = []
    steps = math.floor(60 / interval * 60 * 12)
    for _ in range(steps):
        try:
            _, timestamps, response_timestamp = get_water_data()

            if timestamps not in time_data:
                time_data.append(timestamps)
                time_response.append(response_timestamp)

            df = pd.DataFrame({'time_data': time_data, 'time_response': time_response})
            df.to_csv('./outputs/USGS_iv_api/time_latency.csv', index=False)

        except:
            print('No response.')

        time.sleep(interval)

    print('Finish!')


print()
