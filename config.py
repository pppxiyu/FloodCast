import numpy as np


"""

target_in_forward: int, starting from 1

model: 'naive', 'logistics', 

positive_weight: 'balanced' (only valid if 'logistics' is used for model)
"""


# preprocess and create sequences
threshold_sigma = 4
features = ['water_level', 'discharge']
target = 'surge'
lags = (np.arange(4) + 1).tolist() + [95, 96, 97, 98]
forward = [1]
target_in_forward = 1

# model
model = 'LSTM'
if_weight = True # if weight the imbalanced classes

# cv and hp tune
if_cv = False
if_tune = False
tune_rep_num = 3

# create datasets (work when cv is disabled)
test_percent = 0.15
val_percent = 0.15

# train hp (work when tuning is disabled)
batch_size = 64
learning_rate = 0.003
weight_loss_level_discharge = 3.5 # the weight of the level_discharge loss term

# expr
expr_label = 'LSTM'

random_seed = 0
