import math


def train_pred(df, target_col, forward, target_in_forward, test_percent):
    # USE: use persistence method to pred water level surge
    # INPUT: df
    #        test_index, list, the index of test targets
    # OUTPUT: df, one col is true, another is pred

    # train

    # pred
    test_index = list(range(len(df)))[-math.floor(len(df) * test_percent):]  # test set index
    test_df = df.iloc[test_index][['surge']].copy().rename(columns={'surge': 'true'})

    df['pred'] = df[[target_col]].shift(forward[target_in_forward - 1])
    test_df['pred'] = df['pred'].iloc[test_index]

    return test_df
