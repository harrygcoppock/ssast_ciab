import numpy as np
from scipy import stats
from sklearn import metrics
import torch
import matplotlib.pyplot as plt

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    # unweighted average recall
    uar = metrics.recall_score(np.argmax(target, 1), np.argmax(output, 1), average='macro')

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        dict = {'precisions': precisions,
                'recalls': recalls,
                'AP': avg_precision,
                'fpr': fpr,
                'fnr': 1. - tpr,
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc,
                'uar': uar
                }
        stats.append(dict)

    return stats

