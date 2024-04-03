import pandas as pd
import utils.features as ft
import numpy as np
from sklearn.linear_model import LinearRegression
import utils.preprocess as pp
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

def train_pred(df, df_precip, df_field, adj_matrix_dir, lags, forward, val_percent, test_percent, target_gage):
    # USE: use persistence method to pred water level surge
    # INPUT: df
    #        test_index, list, the index of test targets
    # OUTPUT: df, one col is true, another is pred

    # data
    scaler = PowerTransformer

    df = df.resample('H', closed='right', label='right').mean()
    # df_dis_normed = (df - df.min()) / (df.max() - df.min())
    scaler_stream = scaler()
    df_dis_normed = pd.DataFrame(scaler_stream.fit_transform(df), columns=df.columns, index=df.index)
    dis_cols = [col for col in df.columns if col.endswith('00060')]
    # dis_cols = [f'{target_gage}_00060']
    df_dis_normed = df_dis_normed[dis_cols]
    df_dis_normed = pp.sample_weights(df_dis_normed, f'{target_gage}_00060', if_log=False)

    # adj_precip = pd.read_csv(f'{adj_matrix_dir}/adj_matrix_precipitation.csv', index_col=0)
    area_ratio_precip = pd.read_csv(f'{adj_matrix_dir}/area_in_boundary_ratio.csv')
    area_ratio_precip['lat'] = area_ratio_precip['identifier'].str.split('_').str.get(0)
    area_ratio_precip['lat'] = area_ratio_precip['lat'].astype(float)
    area_ratio_precip['lat'] = area_ratio_precip['lat'] - 0.05
    area_ratio_precip['lon'] = area_ratio_precip['identifier'].str.split('_').str.get(1)
    area_ratio_precip['lon'] = area_ratio_precip['lon'].astype(float)
    area_ratio_precip['lon'] = area_ratio_precip['lon'] - 0.05
    area_ratio_precip['label'] = area_ratio_precip.apply(
        lambda x: f"clat{round(x['lat'], 1)}_clon{round(x['lon'], 1)}",
        axis=1,
    )
    df_precip_scaled = df_precip[area_ratio_precip['label'].to_list()]
    for col in df_precip_scaled.columns:
        df_precip_scaled.loc[:, col] = df_precip_scaled[col] * area_ratio_precip[
            area_ratio_precip['label'] == col
            ]['updated_area_ratio'].iloc[0]
    df_precip_scaled = df_precip_scaled.sum(axis=1).to_frame()
    # df_precip_normed = ((df_precip_scaled - df_precip_scaled.min().min())
    #                     / (df_precip_scaled.max().max() - df_precip_scaled.min().min()))
    scaler_precip = scaler()
    df_precip_normed = pd.DataFrame(
        scaler_precip.fit_transform(df_precip_scaled), columns=df_precip_scaled.columns, index=df_precip_scaled.index
    )
    # df_precip_normed = df_precip_normed[
    #     [f"clat{str(round(float(i.split('_')[0]) - 0.05, 1))}_clon{str(round(float(i.split('_')[1]) - 0.05, 1))}"
    #      for i in adj_precip.columns]
    # ]

    df_normed = pd.concat([
        df_dis_normed,
        df_precip_normed
    ], axis=1)

    target_in_forward = 1
    inputs = ([f'{target_gage}_00060_weights', f'{target_gage}_00060']
              + [col for col in dis_cols if target_gage not in col]
              + list(df_precip_normed.columns)
              )

    # index split and use the index to split data
    df_normed['index'] = range(len(df_normed))
    sequences_w_index = ft.create_sequences(df_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # keep usable field measurements (new)
    start_time = df_normed[df_normed['index'] == sequences_w_index[0,0,-1]].index
    df_field = df_field[df_field.index >= start_time.strftime('%Y-%m-%d %H:%M:%S')[0]]
    if len(df_field) < 50:
        warnings.warn(f'Field measurement count is low. {len(df_field)} usable field visits.')

    # index split for major data
    test_percent_updated, test_df_field, _ = ft.update_test_percent(df_field, df_normed,
                                                                    sequences_w_index, test_percent)
    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]
    dataset_index = ft.create_index_4_cv(x, None, None,
                                         val_percent, test_percent_updated, None,
                                         None)  # test/val percent for cv is not revised

    x = sequences_w_index[:, :, :-1][:, :-len(forward), :]  # hard coded
    y = sequences_w_index[:, :, :-1][:, -len(forward):, :]
    y_index = sequences_w_index[:, :, [-1]][:, -len(forward):, :]

    train_x = x[dataset_index[0]['train_index'], :, :][:, :, 1:]  # hard coded here
    train_y = y[dataset_index[0]['train_index'], :][:, :, :]  # hard coded here
    test_x = x[dataset_index[0]['test_index'], :, :][:, :, 1:]  # hard coded here
    test_y_index = y_index[dataset_index[0]['test_index'], :, 0]

    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

    # model training
    model = LinearRegression()
    model.fit(train_x, train_y[:, 0, 1:], sample_weight=train_y[:, 0, 0])

    # pred
    pred_y = model.predict(test_x)
    pred_y = pred_y[:, 0]
    scaler_pred = scaler()
    scaler_pred.fit(df[[f"{target_gage}_00060"]])
    pred_y = scaler_pred.inverse_transform(pd.DataFrame(pred_y))[:, 0]
    # pred_y = pred_y * (
    #         df[f'{target_gage}_00060'].max() - df[f'{target_gage}_00060'].min()
    # ) + df[f'{target_gage}_00060'].min()

    # modeled discharge
    test_df = df.iloc[test_y_index[:, target_in_forward - 1]][[f'{target_gage}_00060', f'{target_gage}_00065']]
    test_df = test_df.rename(columns={
        f'{target_gage}_00060': 'modeled',
        f'{target_gage}_00065': 'water_level',
    })

    # pred discharge
    test_df['pred'] = pred_y
    test_df_full = test_df.copy()

    # field discharge
    test_df_field.index = test_df_field.index.ceil('H')
    test_df_field = test_df_field.groupby(level=0).mean()
    test_df['field'] = test_df_field['discharge']
    test_df = test_df[~test_df['field'].isna()]

    return test_df, test_df_full
