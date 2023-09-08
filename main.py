import utils.preprocess as pp
import models.naive as naive
import models.logistics as logistics
import models.LSTM as LSTM
import utils.eval as ev
import config

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

# prepare data
data_combined = pp.data_imported_combine(
    ['./data/waterLevel_Columbus.txt', 'water_level'],
    ['./data/discharge_Columbus.txt', 'discharge'],
)
data_combined = pp.data_add_target(data_combined, threshold, forward, col_name='water_level')

# models
if model_name == 'naive':
    test_df = naive.train_pred(data_combined, target_name, forward, target_in_forward, test_percent)

if model_name == 'logistics':
    test_df = logistics.train_pred(data_combined,
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
    test_df = LSTM.train_pred(data_combined,
                              feature_names, target_name, lags, forward,
                              target_in_forward,
                              val_percent, test_percent,
                              if_weight,
                              batch_size,
                              learning_rate,
                              random_seed,
                              )

if model_name == 'PI-LSTM':
    pass

# eval
ev.metric_classi_report(test_df)

print('Running ends.')
