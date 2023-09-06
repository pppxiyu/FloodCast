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
# lags = (np.arange(4) + 1).tolist() + [95, 96, 97, 98]
lags = (np.arange(4) + 1).tolist()
forward = [1, 2]
target_in_forward = 2

# create datasets
test_percent = 0.15
val_percent = 0.15

# model
model = 'logistics'
positive_weight = "balanced"

random_seed = 0
