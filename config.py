import numpy as np


"""

target_in_forward: int, starting from 1

model: 'naive', 'logistics', 

positive_weight: 'balanced' (only valid if 'logistics' is used for model)
"""


# preprocess and create sequences
threshold_sigma = 2
features = ['water_level', 'discharge']
target = 'surge'
lags = (np.arange(4) + 1).tolist() + [95, 96, 97, 98]
forward = [1]
target_in_forward = 1

# create datasets
test_percent = 0.15
val_percent = 0.15

# model
model = 'LSTM'
if_weight = True
batch_size = 256
learning_rate = 0.001

random_seed = 0
