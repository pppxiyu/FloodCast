import utils.features as ft
import numpy as np
import warnings


def train_pred(df, df_field, lags, forward, val_percent, test_percent):
    # USE: use persistence method to pred water level surge
    # INPUT: df
    #        test_index, list, the index of test targets
    # OUTPUT: df, one col is true, another is pred

    df = df.rename(
        columns={
            [col for col in df.columns if col.endswith('00060')][0]: 'discharge',
            [col for col in df.columns if col.endswith('00065')][0]: 'water_level'
        }
    )

    # params
    target_col = 'discharge'
    target_in_forward = 1
    inputs = ['discharge']

    # data
    df = df.resample('H', closed='right', label='right').mean()
    df_normed = (df - df.min()) / (df.max() - df.min())
    df_normed = df_normed[inputs]

    # make sequences and remove samples with nan values
    df_normed['index'] = range(len(df_normed))
    sequences_w_index = ft.create_sequences(df_normed, lags, forward, inputs + ['index'])
    rows_with_nan = np.any(np.isnan(sequences_w_index), axis=(1, 2))
    sequences_w_index = sequences_w_index[~rows_with_nan]

    # keep usable field measurements (new)
    start_time = df_normed[df_normed['index'] == sequences_w_index[0,0,-1]].index
    df_field = df_field[df_field.index >= start_time.strftime('%Y-%m-%d %H:%M:%S')[0]]
    if len(df_field) < 50:
        warnings.warn(f'Field measurement count is low. {len(df_field)} usable field visits.')

    # index split
    test_percent_updated, test_df_field, _ = ft.update_test_percent(df_field, df_normed,
                                                                    sequences_w_index, test_percent)
    dataset_index = ft.create_index_4_cv(sequences_w_index[:, :, :-1][:, :-1, :],
                                         False, None,
                                         val_percent, test_percent_updated, None, None)
    test_y_index = sequences_w_index[:, :, [-1]][:, -1:, :][dataset_index[0]['test_index'], :, 0]

    # modeled discharge
    test_df = df.iloc[test_y_index[:, target_in_forward - 1]][['discharge', 'water_level']]
    test_df = test_df.rename(columns={'discharge': 'modeled'})

    # pred discharge
    df['pred'] = df[[target_col]].shift(forward[target_in_forward - 1])  # hard coded
    test_df['pred'] = df['pred'].iloc[test_y_index[:, 0].tolist()]
    test_df_full = test_df.copy()

    # field discharge
    test_df_field.index = test_df_field.index.ceil('H')
    test_df_field = test_df_field.groupby(level=0).mean()
    test_df['field'] = test_df_field['discharge']
    test_df = test_df[~test_df['field'].isna()]

    return test_df, test_df_full
