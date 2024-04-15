import pandas as pd
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

# preprocess and create sequences
target_gage = gauge_forecast['SITENO'].to_list()
upstream_gages = gauge_forecast['up_gage_names'].to_list()
lags = [[i + 1 for i in list(range(24))]] * len(gauge_forecast)
forward = [[1]] * len(gauge_forecast)

# model
model_name = ['pi_hodcrnn'] * len(gauge_forecast)

# cv and hp tune
if_cv = [False] * len(gauge_forecast)
if_tune = [True] * len(gauge_forecast)

# create datasets (work when cv is disabled)
test_percent = [0.25] * len(gauge_forecast)
val_percent = [0.15] * len(gauge_forecast)

# expr
extra_label = [''] * len(gauge_forecast)
random_seed = [0] * len(gauge_forecast)
