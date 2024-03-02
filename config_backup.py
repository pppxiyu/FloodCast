# preprocess and create sequences
target_gage = ['01573560']
upstream_gages = [["01573160", "01573000", "01572190", "01572025",]]
lags = [[i + 1 for i in list(range(24))]]
forward = [[1]]

# model
model_name = ['linear']

# cv and hp tune
if_cv = [False]
if_tune = [False]

# create datasets (work when cv is disabled)
test_percent = [0.21]
val_percent = [0.15]

# expr
extra_label = ['']
random_seed = [0]
