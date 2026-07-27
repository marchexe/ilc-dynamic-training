import numpy as np
import traceback
import sklearn.metrics as _m
from functools import partial
from ..logger import _logger

# def _bkg_rejection(y_true, y_score, sig_eff):
#     fpr, tpr, _ = _m.roc_curve(y_true, y_score)
#     idx = next(idx for idx, v in enumerate(tpr) if v > sig_eff)
#     rej = 1. / fpr[idx]
#     return rej
#
#
# def bkg_rejection(y_true, y_score, sig_eff):
#     if y_score.ndim == 1:
#         return _bkg_rejection(y_true, y_score, sig_eff)
#     else:
#         num_classes = y_score.shape[1]
#         for i in range(num_classes):
#             for j in range(i + 1, num_classes):
#                 weights = np.logical_or(y_true == i, y_true == j)
#                 truth =


def roc_auc_score_ovo(y_true, y_score):
    if y_score.ndim == 1:
        return _m.roc_auc_score(y_true, y_score)
    else:
        num_classes = y_score.shape[1]
        result = np.zeros((num_classes, num_classes), dtype='float32')
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                weights = np.logical_or(y_true == i, y_true == j)
                truth = y_true == j
                score = y_score[:, j] / np.maximum(y_score[:, i] + y_score[:, j], 1e-6)
                result[i, j] = _m.roc_auc_score(truth, score, sample_weight=weights)
    return result


def _bkg_rejection_for_pair(y_true, y_score, tag_idx, bkg_idx, eff_points):
    selected = np.logical_or(y_true == tag_idx, y_true == bkg_idx)
    if not np.any(selected):
        return np.full(len(eff_points), np.nan, dtype='float32')

    truth = y_true[selected] == tag_idx
    if not np.any(truth) or np.all(truth):
        return np.full(len(eff_points), np.nan, dtype='float32')

    scores = y_score[selected, tag_idx] / np.maximum(
        y_score[selected, tag_idx] + y_score[selected, bkg_idx], 1e-6
    )
    fpr, tpr, _ = _m.roc_curve(truth, scores)
    min_bkg_eff = 1.0 / (np.count_nonzero(~truth) + 1.0)
    rejection = []
    for eff in eff_points:
        passing = np.flatnonzero(tpr >= eff)
        if not passing.size:
            rejection.append(np.nan)
            continue
        bkg_eff = max(fpr[passing[0]], min_bkg_eff)
        rejection.append(1.0 / bkg_eff)
    return np.asarray(rejection, dtype='float32')


def bkg_rejection_at_eff(y_true, y_score):
    """Background rejection curves for b/c/d pairwise flavour comparisons.

    The labels are expected to follow the local SGV 3-class order:
    0=b, 1=c, 2=d.  Pair names use <tag><background>, e.g. ``bc`` means
    b-tag efficiency on the x-axis and c-background rejection on the y-axis.
    """

    if y_score.ndim == 1 or y_score.shape[1] < 3:
        raise ValueError('bkg_rejection_at_eff requires b/c/d multiclass scores')

    eff_points = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype='float32')
    pairs = {
        'bc': (0, 1),
        'bd': (0, 2),
        'cb': (1, 0),
        'cd': (1, 2),
    }
    return {
        pair: _bkg_rejection_for_pair(y_true, y_score, tag_idx, bkg_idx, eff_points).tolist()
        for pair, (tag_idx, bkg_idx) in pairs.items()
    }


def _bkg_rejection_score_from_pairs(y_true, y_score, pair_names):
    curves = bkg_rejection_at_eff(y_true, y_score)
    values = np.asarray(
        [value for pair in pair_names for value in curves[pair]],
        dtype='float64',
    )
    finite = values[np.isfinite(values) & (values > 0)]
    if not finite.size:
        return np.nan
    return float(np.mean(np.log(finite)))


def bkg_rejection_bc_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('bc',))


def bkg_rejection_bd_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('bd',))


def bkg_rejection_cb_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('cb',))


def bkg_rejection_cd_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('cd',))


def b_tag_rejection_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('bc', 'bd'))


def c_tag_rejection_score(y_true, y_score):
    return _bkg_rejection_score_from_pairs(y_true, y_score, ('cb', 'cd'))


def bkg_rejection_score(y_true, y_score):
    """Scalar PBT ranking score from all bc/bd/cb/cd bkg rejection curves.

    Uses the mean log(background rejection) across fixed signal efficiency
    points.  Log-space keeps one easy pair from dominating the rank.
    """

    return _bkg_rejection_score_from_pairs(y_true, y_score, ('bc', 'bd', 'cb', 'cd'))


def confusion_matrix(y_true, y_score):
    if y_score.ndim == 1:
        y_pred = y_score > 0.5
    else:
        y_pred = y_score.argmax(1)
    return _m.confusion_matrix(y_true, y_pred, normalize='true')


_metric_dict = {
    'roc_auc_score': partial(_m.roc_auc_score, multi_class='ovo'),
    'roc_auc_score_matrix': roc_auc_score_ovo,
    'bkg_rejection_at_eff': bkg_rejection_at_eff,
    'bkg_rejection_bc_score': bkg_rejection_bc_score,
    'bkg_rejection_bd_score': bkg_rejection_bd_score,
    'bkg_rejection_cb_score': bkg_rejection_cb_score,
    'bkg_rejection_cd_score': bkg_rejection_cd_score,
    'b_tag_rejection_score': b_tag_rejection_score,
    'c_tag_rejection_score': c_tag_rejection_score,
    'bkg_rejection_score': bkg_rejection_score,
    'confusion_matrix': confusion_matrix,
}


def _get_metric(metric):
    try:
        return _metric_dict[metric]
    except KeyError:
        return getattr(_m, metric)


def evaluate_metrics(y_true, y_score, eval_metrics=[]):
    results = {}
    for metric in eval_metrics:
        if callable(metric):
            metric, func = metric.__name__, metric
        else:
            func = _get_metric(metric)
        try:
            results[metric] = func(y_true, y_score)
        except Exception as e:
            results[metric] = None
            _logger.warning(f'Cannot compute metric {metric}: {str(e)}')
            _logger.debug(traceback.format_exc())
    return results
