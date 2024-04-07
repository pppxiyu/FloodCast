import utils.preprocess as pp
import utils.eval as ev

import config

import os
import json
from datetime import datetime

import models.baselines.naive as naive
import models.baselines.linear as linear
import models.baselines.mlp as mlp
import models.baselines.xgboost as xgboost
import models.baselines.gru as gru

import models.hodcrnn as hodcrnn
import models.pending.hodcrnn_tune_o as hodcrnn_tune_o
import models.pending.hodcrnn_tune_h as hodcrnn_tune_h

import models.pi_hodcrnn as pi_hodcrnn
import models.pi_hodcrnn_tune_base as pi_hodcrnn_tune_base
import models.pi_hodcrnn_tune_o1 as pi_hodcrnn_tune_o1
import models.pi_hodcrnn_tune_o2 as pi_hodcrnn_tune_o2
import models.pi_hodcrnn_tune_o3 as pi_hodcrnn_tune_o3
import models.pending.pi_hodcrnn_tune_lump as pi_hodcrnn_tune_l

import pandas as pd


# config
target_gage_list = config.target_gage
upstream_gages_list = config.upstream_gages
lags_list = config.lags
forward_list = config.forward
model_name_list = config.model_name
extra_label_list = config.extra_label
if_tune_list = config.if_tune
if_cv_list = config.if_cv
test_percent_list = config.test_percent
val_percent_list = config.val_percent

dir_cache = './data/cache'

for target_gage, upstream_gages, lags, forward, model_name, extra_label, if_tune, if_cv, test_percent, val_percent in zip(
        target_gage_list, upstream_gages_list, lags_list, forward_list, model_name_list, extra_label_list,
        if_tune_list, if_cv_list, test_percent_list, val_percent_list
):
    if target_gage == '01573560':
        print(f'Forecasting for {target_gage}.')
        # revise gauge and up gauge
        with open('./outputs/USGS_gaga_filtering/gauge_upstream_delete.json', 'r') as f:
            remove_dict = json.load(f)
        with open('./outputs/USGS_gaga_filtering/gauge_delete.json', 'r') as f:
            remove_list = json.load(f)
        with open('./outputs/USGS_gaga_filtering/gauge_delete_action_during_test_025.json', 'r') as f:
            remove_list_2 = json.load(f)
        if target_gage in remove_list + ['04293500']:
            continue
        if target_gage in remove_list_2:
            continue
        if target_gage in list(remove_dict.keys()):
            upstream_gages = [i for i in upstream_gages if i not in remove_dict[target_gage]]

        # create experiment
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        experiment_label = model_name + extra_label + '_' + str(current_time)
        expr_dir = f'./outputs/experiments/{model_name}_{str(forward[0])}_{extra_label}_{str(current_time)}'
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        config_info_dict = {
            'target_gage': target_gage, 'upstream_gages': upstream_gages,
            'lags': lags, 'forward': forward,
            'model_name': model_name, 'extra_label': extra_label,
            'if_tune': if_tune, 'if_cv': if_cv,
            'test_percent': test_percent, 'val_percent': val_percent
        }
        with open(f'{expr_dir}/config.json', 'w') as f:
            json.dump(config_info_dict, f)

        # prepare data
        if os.path.isfile(f'{dir_cache}/data_{target_gage}.csv'):
            data = pd.read_csv(f'{dir_cache}/data_{target_gage}.csv', index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, utc=True)
            data.index = data.index.tz_convert('America/New_York')
            # check extra col
            extra_col = [g for g in list(set(
                [i.split('_')[0] for i in data.columns]
            )) if g not in upstream_gages + [target_gage]]
            extra_col_delete = [col for col in data.columns if col.split('_')[0] in extra_col]
            data = data.drop(extra_col_delete, axis=1)
        else:
            data = pp.import_data_combine(
                [f'./data/USGS_gage_iv_20y/{gage}.csv' for gage in upstream_gages + [target_gage]],
                tz='America/New_York',
                keep_col=['00065', '00060']
            )
            data.to_csv(f'{dir_cache}/data_{target_gage}.csv')

        # if os.path.isfile(f'{dir_cache}/data_precip_{target_gage}.csv'):
        #     data_precip = pd.read_csv(f'{dir_cache}/data_precip_{target_gage}.csv', index_col=0, parse_dates=True)
        #     data_precip.index = pd.to_datetime(data_precip.index, utc=True)
        #     data_precip.index = data_precip.index.tz_convert('America/New_York')
        # else:
        #     with open(f'./data/USGS_basin_geo/{target_gage}_basin_geo.geojson', 'r') as f:
        #         watershed = json.load(f)
        #     b_lat_min, b_lat_max, b_lon_min, b_lon_max = pp.get_bounds(watershed)
        #     lat_list, lon_list = [f'{b_lat_max}'], [f'{b_lon_min}']
        #     data_precip = pp.import_data_precipitation_legacy(
        #         './data/JAXA_precipitation_data/concatenated',
        #         lat_list, lon_list,
        #         'America/New_York'
        #     )
        #     data_precip.to_csv(f'{dir_cache}/data_precip_{target_gage}.csv')
        data_precip = pp.import_data_precipitation(
            './data/JAXA_precipitation_data/concatenated',
            target_gage,
            'America/New_York'
        )

        data_field = pp.import_data_field(f'./data/USGS_gage_field/{target_gage}.csv', to_tz='America/New_York')

        data_rc = pp.import_data_rc(f'./data/USGS_gage_rc/{target_gage}_rc.txt')
        data_flood_stage = pd.read_csv(f'./data/USGS_gage_flood_stage/flood_stages.csv', dtype={'site_no': 'str'})
        data_flood_stage = data_flood_stage[data_flood_stage['site_no'] == target_gage]

        adj_matrix_dir = f'./outputs/USGS_{target_gage}/adj_matrix_USGS_{target_gage}'

        # modeling and prediction
        if model_name == 'naive':
            test_df, test_df_full = naive.train_pred(
                data[[f'{target_gage}_00060', f'{target_gage}_00065']],
                data_field, lags, forward, val_percent, test_percent
            )
        if model_name == 'linear':
            test_df, test_df_full = linear.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, val_percent, test_percent, target_gage
            )
        if model_name == 'mlp':
            test_df, test_df_full = mlp.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                if_tune=if_tune,
            )
        if model_name == 'xgboost':
            test_df, test_df_full = xgboost.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                if_tune=if_tune,
            )

        if model_name == 'gru':
            test_df, test_df_full = gru.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                if_tune=if_tune,
            )

        if model_name == 'hodcrnn':
            test_df, test_df_full = hodcrnn.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir,
                if_tune=if_tune,
            )

        if model_name == 'hodcrnn_tune_o':
            test_df, test_df_full = hodcrnn_tune_o.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir,
                if_tune=if_tune,)

        if model_name == 'hodcrnn_tune_h':
            test_df, test_df_full = hodcrnn_tune_h.train_pred(
                data, data_precip, data_field,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir,
                if_tune=if_tune,)

        if model_name == 'pi_hodcrnn':
            test_df, test_df_full = pi_hodcrnn.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir, data_flood_stage,
                if_tune=if_tune,
            )

        if model_name == 'pi_hodcrnn_tune_base':
            test_df, test_df_full = pi_hodcrnn_tune_base.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir, data_flood_stage,
                if_tune=if_tune,
            )

        if model_name == 'pi_hodcrnn_tune_o1':
            test_df, test_df_full = pi_hodcrnn_tune_o1.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir, data_flood_stage,
                if_tune=if_tune,
            )

        if model_name == 'pi_hodcrnn_tune_o2':
            test_df, test_df_full = pi_hodcrnn_tune_o2.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir, data_flood_stage,
                if_tune=if_tune,
            )

        if model_name == 'pi_hodcrnn_tune_o3':
            test_df, test_df_full = pi_hodcrnn_tune_o3.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir, data_flood_stage,
                if_tune=if_tune,
            )

        if model_name == 'pi_hodcrnn_tune_l':
            test_df, test_df_full = pi_hodcrnn_tune_l.train_pred(
                data, data_precip, data_field, data_rc,
                adj_matrix_dir, lags, forward, target_gage, val_percent, test_percent,
                expr_dir,
                if_tune=if_tune,
            )

        if 'test_df' in locals():
            pd.set_option('display.max_columns', None)
            report_df = ev.metric_dis_pred_report(
                test_df, test_df_full, data_flood_stage['action'].iloc[0], target_gage,
            )

            # save
            test_df.to_csv(expr_dir + '/' + 'test_df.csv')
            test_df_full.to_csv(expr_dir + '/' + 'test_df_full.csv')
            report_df.to_csv(expr_dir + '/report_df.csv', index=True)

print('Running ends.')
