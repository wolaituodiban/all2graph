from sklearn.metrics import roc_curve


def ks_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)


