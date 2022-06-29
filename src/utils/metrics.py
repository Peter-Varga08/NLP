from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from enums import MetricType

ClfReports = List[Tuple[NDArray]]
AccuracyScores = List[float]
ExplainedClfReport = Dict[int, Dict[str, float]]


def get_avg_clf_report(clf_report: ClfReports) -> NDArray:
    return np.round(np.mean(clf_report, axis=0), 4)


def get_avg_accuracy(accuracy_scores: AccuracyScores) -> float:
    return np.round(np.mean(accuracy_scores), 4)


def get_avg_score(scores: Dict[MetricType, Union[ClfReports, AccuracyScores]]):
    """
    Return average of each passed score. Requires 'clf_report' and 'accuracy_score' keys.
    """
    avg_score = {}
    for metric in scores:
        if metric is MetricType.CLF_REPORT:
            avg_score[metric] = get_avg_clf_report(scores[metric])
        elif metric is MetricType.ACCURACY_SCORE:
            avg_score[metric] = get_avg_accuracy(scores[metric])
        else:
            raise TypeError("Wrong metric type used as key in input.")
    return avg_score


def explain_clf_score(score: NDArray) -> ExplainedClfReport:
    """
    Add 'avg_precision', 'avg_recall', 'avg_f1' 'avg_support', string literals as keys to the raw score values
    from a precision_recall_fscore_support() call on the K-Fold CV result of a classifier.
    """
    return {
        str(idx): {
            "avg_precision": row[0],
            "avg_recall": row[1],
            "avg_f1": row[2],
            "avg_support": support,
        }
        for idx, (row, support) in enumerate(zip(score[:-1], score[-1]))
    }
