import numpy as np


"""

target_in_forward: int, starting from 1

model: 'naive', 'logistics', 

positive_weight: 'balanced' (only valid if 'logistics' is used for model)
"""


# preprocess and create sequences
threshold_sigma = 3
features = ['water_level', 'discharge']
target = 'surge'
lags = (np.arange(4) + 1).tolist() + [95, 96, 97, 98]
forward = [1]
target_in_forward = 1

# model
model = 'LSTM'
if_weight = True

# cv and hp tune
if_cv = False
if_tune = True
tune_rep_num = 1

# create datasets (work when cv is disabled)
test_percent = 0.15
val_percent = 0.15

# train hp (work when tuning is disabled)
batch_size = 128
learning_rate = 0.005

# expr
expr_label = 'LSTM'

random_seed = 0
