from sklearn.metrics import classification_report


def metric_classi_report(df):
    # USE: print classification report
    # INPUT: test_results df
    # OUTPUT:
    true = df['true']
    pred = df['pred']
    target_names = ['No flooding', 'Flooding']
    print(classification_report(true, pred, target_names=target_names))

    return
