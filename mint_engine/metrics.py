import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def manual_auc(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    cursor = 0
    while cursor < len(scores):
        end = cursor
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[cursor]:
            end += 1
        midpoint = ((cursor + 1) + (end + 1)) / 2.0
        ranks[cursor:end + 1] = midpoint
        cursor = end + 1
    positive = np.where(sorted_labels == 1.0)[0]
    negative = np.where(sorted_labels == 0.0)[0]
    n_positive = len(positive)
    n_negative = len(negative)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def manual_aupr(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    n_positive = max(1, int(labels.sum()))
    accumulated = 0.0
    previous_recall = 0.0
    previous_precision = 1.0
    for value in sorted_labels:
        if value == 1.0:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / n_positive
        accumulated += (recall - previous_recall) * (previous_precision + precision) / 2.0
        previous_recall = recall
        previous_precision = precision
    return float(accumulated)


def metric_block(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    block = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
    }
    try:
        block["auc"] = float(roc_auc_score(labels, scores))
    except Exception:
        block["auc"] = float("nan")
    try:
        block["manual_auc"] = manual_auc(labels, scores)
    except Exception:
        block["manual_auc"] = float("nan")
    try:
        block["aupr"] = float(average_precision_score(labels, scores))
    except Exception:
        block["aupr"] = float("nan")
    block["manual_aupr"] = manual_aupr(labels, scores)
    block["accuracy"] = float(accuracy_score(labels, predicted))
    return block


def agreement_check(first, second, tolerance=1e-9):
    if first != first or second != second:
        return False
    return abs(first - second) <= tolerance


def verify_metrics(block, auc_tolerance=1e-9, aupr_tolerance=1e-2):
    return {
        "auc": agreement_check(block["auc"], block["manual_auc"], auc_tolerance),
        "aupr": agreement_check(block["aupr"], block["manual_aupr"], aupr_tolerance),
    }