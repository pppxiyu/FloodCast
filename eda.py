import pandas as pd
import os

import utils.preprocess as pp


analysis_name = 'check_duplicates&missing'
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
    import utils.vis as vis

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
    import plotly.express as px
    import plotly.graph_objects as go

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

if analysis_name == 'pull_JAXA_data_point':

    import os
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import json
    import math
    from ast import literal_eval

    def pull_jaxa():  # this function is not up to date
        gauge_forecast = pd.read_csv(
            "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
            dtype={"SITENO": str},
        )
        gauge_forecast['up_gage_names'] = gauge_forecast.apply(
            lambda row: sorted(list(set(
                literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
                )), reverse=True), axis=1
        )
        # temporary selection
        gauge_forecast = gauge_forecast[
            (gauge_forecast['active_up_gage_num_main'] >= 3)
            & (gauge_forecast['field_measure_count_action'] >= 10)
            ]

        uid = 'rainmap'
        psd = 'Niskur+1404'

        working_dir = 'C:/Users/xpan88/Downloads'
        for gg in gauge_forecast['SITENO'].to_list():
            print(f'Downloading for {gg}.')
            download_flag = False
            dts = pd.date_range(
                start='1/1/2007',
                end='01/01/2024',
                freq='4MS',
                tz='America/New_York'  # tz is not customized for each gauge
            ).tz_convert('UTC').strftime('%Y%m%d%H').to_list()

            with open(f'./data/USGS_basin_geo/{gg}_basin_geo.geojson', 'r') as f:
                watershed = json.load(f)
            lat_list, lon_list = pp.get_bounding_grid(watershed)

            st_list = dts[:-1]
            ed_list = dts[1:]

            csv_files = [file for file in os.listdir(working_dir) if file.endswith('.csv')]
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
                            download_flag = True
                            break
                    if initial:
                        break
                if initial:
                    break
            if not initial:
                # raise EOFError('All files have been pulled.')
                continue

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

            if download_flag is not True:
                print(f'Downloading for {gg} is done.')

            time.sleep(10)
            driver.quit()
        return

    for attempt in range(100):
        try:
            pull_jaxa()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Waiting {60} seconds before retrying...")
            time.sleep(60)
    raise Exception("All attempts failed")
    # pull_jaxa()

if analysis_name == 'pull_JAXA_data_block':

    import os
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import json
    import math
    from ast import literal_eval

    def pull_jaxa():
        gauge_forecast = pd.read_csv(
            "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
            dtype={"SITENO": str},
        )
        gauge_forecast['up_gage_names'] = gauge_forecast.apply(
            lambda row: sorted(list(set(
                literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
                )), reverse=True), axis=1
        )

        uid = 'rainmap'
        psd = 'Niskur+1404'

        working_dir = 'C:/Users/xpan88/Downloads'
        initial = False
        first_run = True
        for gg in gauge_forecast['SITENO'].to_list():
            print(f'Downloading for {gg}.')

            with open(f'./data/USGS_basin_geo/{gg}_basin_geo.geojson', 'r') as f:
                watershed = json.load(f)
            b_lat_min, b_lat_max, b_lon_min, b_lon_max = pp.get_bounds(watershed)

            dts = pd.date_range(
                start='1/1/2007',
                end='01/01/2024',
                freq='4MS',
                tz='America/New_York'  # tz is not customized for each gauge
            ).tz_convert('UTC').strftime('%Y%m%d%H').to_list()
            st_list = dts[:-1]
            ed_list = dts[1:]

            csv_files = [file for file in os.listdir(working_dir) if file.endswith('.csv')]
            saved_file = [(
                i[i.find('_st') + 3: i.find('_ed')],
                i[i.find('_ed') + 3: i.find('_lat')],
                i[i.find('_lat') + 4: i.find('_lon')],
                i[i.find('_lon') + 4: i.find('.csv')],
            ) for i in csv_files]
            saved_file = [(
                i[0], i[1],
                i[3].split('_')[0],
                i[2].split('_')[0],
                i[2].split('_')[1],
                i[3].split('_')[1],
            ) for i in saved_file]

            if not initial:
                for st, ed in zip(st_list, ed_list):
                    if (st, ed, str(b_lat_min), str(b_lat_max), str(b_lon_min), str(b_lon_max)) not in saved_file:
                        # open webpage
                        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                        head = 'https://sharaku.eorc.jaxa.jp/cgi-bin/trmm/GSMaP/tilemap/show_graph.cgi?flag=2&'

                        url = (f"{head}st={st}&ed={ed}"
                               f"&lat0={b_lat_max}&lon0={b_lon_min}&lat1={b_lat_min}&lon1={b_lon_max}"
                               f"&lang=en"
                               )
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

                        saved_file.append((st, ed, b_lat_min, b_lat_max, b_lon_min, b_lon_max))
                        initial = True
                        break
            if not initial:
                continue

            for st, ed in zip(st_list, ed_list):
                if (st, ed, str(b_lat_min), str(b_lat_max), str(b_lon_min), str(b_lon_max)) not in saved_file:
                    if first_run:
                        first_run = False
                        continue
                    url = (f"{head}st={st}&ed={ed}"
                           f"&lat0={b_lat_max}&lon0={b_lon_min}&lat1={b_lat_min}&lon1={b_lon_max}"
                           f"&lang=en"
                           )
                    driver.get(url)
                    button = driver.find_element(By.ID, 'graph_dl')
                    button.click()

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

        time.sleep(5)
        driver.quit()

        return

    # for attempt in range(100):
    #     try:
    #         pull_jaxa()
    #     except Exception as e:
    #         print(f"Attempt {attempt + 1} failed with error: {e}. Waiting {60} seconds before retrying...")
    #         time.sleep(60)
    # raise Exception("All attempts failed")
    pull_jaxa()

if analysis_name == 'check_duplicates&missing':
    import os
    import pandas as pd
    import time
    import json
    import math
    from ast import literal_eval

    gauge_forecast = pd.read_csv(
        "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
        dtype={"SITENO": str},
    )
    gauge_forecast['up_gage_names'] = gauge_forecast.apply(
        lambda row: sorted(list(set(
            literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
        )), reverse=True), axis=1
    )

    working_dir = 'C:/Users/xpan88/Downloads'
    duplicates = []
    for gg in gauge_forecast['SITENO'].to_list():
        with open(f'./data/USGS_basin_geo/{gg}_basin_geo.geojson', 'r') as f:
            watershed = json.load(f)
        b_lat_min, b_lat_max, b_lon_min, b_lon_max = pp.get_bounds(watershed)

        dts = pd.date_range(
            start='1/1/2007',
            end='01/01/2024',
            freq='4MS',
            tz='America/New_York'  # tz is not customized for each gauge
        ).tz_convert('UTC').strftime('%Y%m%d%H').to_list()
        st_list = dts[:-1]
        ed_list = dts[1:]

        csv_files = [file for file in os.listdir(working_dir) if file.endswith('.csv')]
        saved_file = [(
            i[i.find('_st') + 3: i.find('_ed')],
            i[i.find('_ed') + 3: i.find('_lat')],
            i[i.find('_lat') + 4: i.find('_lon')],
            i[i.find('_lon') + 4: i.find('.csv')],
        ) for i in csv_files]
        saved_file = [(
            i[0], i[1],
            i[3].split('_')[0],
            i[2].split('_')[0],
            i[2].split('_')[1],
            i[3].split('_')[1],
        ) for i in saved_file]

        for st, ed in zip(st_list, ed_list):
            occur = saved_file.count( (st, ed, str(b_lat_min), str(b_lat_max), str(b_lon_min), str(b_lon_max)) )
            if occur == 0:
                print('Missing data')
            elif occur >= 2:
                duplicates.append( (st, ed, str(b_lat_min), str(b_lat_max), str(b_lon_min), str(b_lon_max)) )

    if duplicates:
        for du in duplicates:
            csv_du = [
                i for i in csv_files if i[18: ] == f'st{du[0]}_ed{du[1]}_lat{du[3]}_{du[4]}_lon{du[2]}_{du[5]}.csv'
            ]
            assert len(csv_du) >= 2, 'A duplicate is missing.'
            for c in csv_du[1:]:
                os.remove(f'{working_dir}/{c}')



if analysis_name == 'organize_JAXA_data':

    import os
    import pandas as pd
    from ast import literal_eval
    import json

    gauge_forecast = pd.read_csv(
        "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
        dtype={"SITENO": str},
    )
    gauge_forecast['up_gage_names'] = gauge_forecast.apply(
        lambda row: sorted(list(set(
            literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
            )), reverse=True), axis=1
    )
    # temporary selection
    gauge_forecast = gauge_forecast[
        (gauge_forecast['active_up_gage_num_main'] >= 3)
        & (gauge_forecast['field_measure_count_action'] >= 10)
        ]

    working_dir = './data/JAXA_precipitation_data'

    # for gg in gauge_forecast['SITENO']:
    for gg in ['01573560']:

        with open(f'./data/USGS_basin_geo/{gg}_basin_geo.geojson', 'r') as f:
            watershed = json.load(f)
        lat_list, lon_list = pp.get_bounding_grid(watershed)
        needed_loc_list = [(lat, lon) for lat in lat_list for lon in lon_list]

        csv_files = [file for file in os.listdir(working_dir) if file.endswith('.csv')]
        saved_file = [(
            i[i.find('out') + 3: i.find('_st')],
            i[i.find('_st') + 3: i.find('_ed')],
            i[i.find('_ed') + 3: i.find('_clat')],
            i[i.find('_clat') + 5: i.find('_clon')],
            i[i.find('_clon') + 5: i.find('.csv')],
        ) for i in csv_files]
        loc_list = list(set([i[3:5] for i in saved_file]))

        # check if data of all needed locs were downloaded
        if not all(l in loc_list for l in needed_loc_list):
            print(f'Not all data for gauge {gg} is collected')

        concatenated_csv_files = [file for file in os.listdir(f'{working_dir}/USGS_{gg}') if file.endswith('.csv')]
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
                assert ff.index.to_list() == pd.date_range(
                    start=ff.index.min(), end=ff.index.max(), freq='H'
                ).to_list(), 'Datetime index is not consecutive or consistent!'

                ff.to_csv(f'{working_dir}/USGS_{gg}/clat{loc[0]}_clon{loc[1]}.csv')



if analysis_name == 'line_rating_curve':
    import torch
    import utils.modeling as mo
    import plotly.express as px

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


if analysis_name == 'identify_flood_events':
    import matplotlib.pyplot as plt
    import os

    gage = '01573560'
    data_dir = './data/USGS_gage_iv_20y/01573560.csv'
    flood_stage_dir = './data/USGS_gage_flood_stage/flood_stages.csv'
    save_dir = './outputs/USGS_01573560/flooding_period'
    os.makedirs(save_dir, exist_ok=True)
    test_time = ('2021-01-13 11:00:00', '2024-01-01 00:00:00')

    data = pp.import_data(data_dir, tz='America/Chicago')
    data = data.resample('H', closed='right', label='right').mean()
    data = data[(data.index >= test_time[0]) & (data.index <= test_time[1])]

    flood_stage = pd.read_csv(flood_stage_dir, dtype={'site_no': 'str'})
    flood_stage = flood_stage[flood_stage['site_no'] == gage]

    # vis
    data[f'{gage}_00065'].plot()
    plt.axhline(y=flood_stage['action'].values[0], color='r', linestyle='--')
    plt.savefig(f'{save_dir}/action_period.png')
    plt.show()

    # start and end time
    data_action = data[data[f'{gage}_00065'] > flood_stage['action'].values[0]]
    data_action = data_action.reset_index()
    data_action['time_diff'] = data_action['index'].diff()
    flood_starts = data_action.loc[
        (data_action['time_diff'] != pd.Timedelta(hours=1)) | (data_action.index == data_action.index[0]), 'index'
    ].reset_index()['index']
    flood_ends = data_action.loc[
        (data_action['time_diff'].shift(-1) != pd.Timedelta(hours=1)) | (data_action.index == data_action.index[-1]), 'index'
    ].reset_index()['index']
    flood_periods = pd.DataFrame({'start': flood_starts, 'end': flood_ends})

    # peak time
    peak_discharge, peak_time = [], []
    for s, e in zip(flood_starts, flood_ends):
        one_action = data_action[(data_action['index'] > s) & (data_action['index'] < e)]
        max_dis = one_action[f'{gage}_00060'].max()
        peak_discharge.append(max_dis)
        peak_time.append(one_action[one_action[f'{gage}_00060'] == max_dis]['index'].iloc[0])
    flood_periods['peak_dis'] = peak_discharge
    flood_periods['peak_time'] = peak_time

    flood_periods['data_avail'] = [True] * len(flood_periods)  # assume data is avail as default
    flood_periods.to_csv(f'{save_dir}/action_period.csv', index=False)

print()
