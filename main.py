import utils.preprocess as pp
import models.naive as naive
import models.logistics as logistics
import models.LSTM as LSTM
import models.PILSTM as PILSTM
import utils.eval as ev
import config

import os
from datetime import datetime
import json

# config
model_name = config.model
test_percent = config.test_percent
val_percent = config.val_percent
threshold = config.threshold_sigma
if_weight = config.if_weight
feature_names = config.features
target_name = config.target
lags = config.lags
forward = config.forward
target_in_forward = config.target_in_forward
random_seed = config.random_seed
batch_size = config.batch_size
learning_rate = config.learning_rate
if_tune = config.if_tune
expr_label = config.expr_label
if_cv=config.if_cv
tune_rep_num = config.tune_rep_num

# create experiment
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
experiment_label = expr_label + '_' + str(current_time)
expr_dir = './outputs/experiments/' + expr_label + '_' + str(current_time)
if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

with open("./config.py", 'rb') as src_file:
    with open(expr_dir + '/' + "./config.py", 'wb') as dst_file:
        dst_file.write(src_file.read())


# prepare data
data_raw = pp.data_imported_combine(
    ['./data/waterLevel_Columbus.txt', 'water_level'],
    ['./data/discharge_Columbus.txt', 'discharge'],
)
data = data_raw[feature_names]
data = pp.data_add_target(data, threshold, forward, target_name='water_level')

# models
if model_name == 'naive':
    test_df = naive.train_pred(data, target_name, forward, target_in_forward, test_percent)

if model_name == 'logistics':
    test_df = logistics.train_pred(data,
                                   feature_names, target_name, lags, forward,
                                   target_in_forward,
                                   val_percent, test_percent,
                                   if_weight,
                                   random_seed,
                                   )

if model_name == 'light_boost':
    pass

if model_name == 'ann':
    pass

if model_name == 'LSTM':
    test_df = LSTM.train_pred(data,
                              feature_names, target_name, lags, forward,
                              target_in_forward,
                              val_percent, test_percent,
                              if_weight,
                              batch_size,
                              learning_rate,
                              if_tune=if_tune,
                              tune_rep_num=tune_rep_num,
                              if_cv=if_cv,
                              )

if model_name == 'PI-LSTM':
    test_df = LSTM.train_pred(data,
                              feature_names, target_name, lags, forward,
                              target_in_forward,
                              val_percent, test_percent,
                              if_weight,
                              batch_size,
                              learning_rate,
                              if_tune=if_tune,
                              tune_rep_num=tune_rep_num,
                              if_cv=if_cv,
                              )

# eval
report_dict = ev.metric_classi_report(test_df)

# save
test_df.to_csv(expr_dir + '/' + 'test_df.csv')
with open(expr_dir + '/report_dict.json', 'w') as file:
    json.dump(report_dict, file)

print('Running ends.')
