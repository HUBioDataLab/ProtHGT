from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, matthews_corrcoef

def metrics(prob_pred, binary_pred, ground_truth):

    # F1
    f1 = f1_score(ground_truth, binary_pred)

    # PR-AUC
    _precision, _recall, thresholds = precision_recall_curve(ground_truth, prob_pred)
    pr_auc = auc(_recall, _precision)

    # accuracy
    acc = accuracy_score(ground_truth, binary_pred)

    # precision - recall
    precision = precision_score(ground_truth, binary_pred)
    recall = recall_score(ground_truth, binary_pred)

    mcc= matthews_corrcoef(ground_truth, binary_pred)

    score_list = [precision, recall, f1, pr_auc, acc, mcc]

    return [float(score) for score in score_list]