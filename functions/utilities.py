import argparse
import numpy as np
import math
from sklearn import metrics

def getPrecisionAtFixedRecall(y_labels, scores,fixedRecall = 0.95):
    precisions, recalls, _ = metrics.precision_recall_curve(y_labels, scores)
    idxToMatch = -1
    # The following is true because recalls is sorted from largest to smallest
    for idx,recall in enumerate(recalls):
        if recall > fixedRecall:
            continue
        elif recall == fixedRecall:
            try:
                if recalls[idx+1] == fixedRecall:
                    return precisions[idx+1]
            finally:
                return precisions[idx]
        else:
            idxToMatch = idx
            break
    if idxToMatch == -1:
        raise Exception("Found no appropriate recall")
    else:
        return np.interp(x= fixedRecall, xp = [recalls[idxToMatch], recalls[idxToMatch-1]], fp = [precisions[idxToMatch], precisions[idxToMatch-1]])

def valid_positive_int(value):
    """Check if value is positive integer."""
    value = int(value)
    if value < 0 or value > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "must be any integer between 0 and 2**32 - 1 inclusive")
    return value

def valid_strictly_positive_int(value):
    """Check if value is positive integer."""
    value = int(value)
    if value <= 0 or value > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "must be any integer between 1 and 2**32 - 1 inclusive")
    return value

def valid_percentage(value):
    """Check if value is a valid percentage."""
    value = float(value)
    if value >= 1 or value <= 0:
        raise argparse.ArgumentTypeError(
                "must be any integer between 1 and 2**32 - 1 inclusive")
    return value

def convert_label_to_binary(dataset_name, label_encoder, labels):
    if dataset_name == "credit":
        return labels
    elif dataset_name == 'kdd':
        normal_idx = np.where(label_encoder.classes_ == 'normal.')[0][0]
        my_labels = labels.copy()
        my_labels[my_labels != normal_idx] = 1
        my_labels[my_labels == normal_idx] = 0
        return my_labels

def valid_bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def f_measure(recall, precision, beta=1):
    return (1+beta**2)*(precision*recall)/(((beta**2)*precision)+recall)


def compress(arr, eps):
    min_val = np.min(arr[np.nonzero(arr)])
    max_val = np.max(arr[np.nonzero(arr)])
    k = int(math.log(max_val / min_val) / math.log(1 + eps))
    sp_vals = sorted({min_val * ((1 + eps) ** i) for i in range(k + 1)})
    sp_indices = sorted(set(np.searchsorted(arr, sp_vals)))
    compress_arr = {index: arr[index] for index in sp_indices}
    return compress_arr

