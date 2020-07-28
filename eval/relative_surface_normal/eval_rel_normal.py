'''
  Written by Weifeng Chen
  Reviewed by Noriyuki Kojima 04/10/2019, 04/24/2019, 04/29/2019
  Reviewed by Weifeng Chen 04/24/2019
  Modified by Noriyuki Kojima 05/18/2019
  Reviewed by Weifeng Chen 05/19/2019
'''
from sklearn.metrics import auc
from numpy import linalg
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# from IPython import embed

# normalize normals
def normalize(normal_map):
    l2_norms = linalg.norm(normal_map, axis = 2)
    l2_norms = np.stack([l2_norms, l2_norms, l2_norms], axis = -1)
    return np.divide(normal_map, l2_norms)


def ang_diff(vec1, vec2):
    '''
        calculate the angle difference between two vectors
        return between 0 and pi, in radians
    '''
    dot_prod = np.dot(vec1, vec2)
    dot_prod = min(dot_prod, 1.0)
    dot_prod = max(dot_prod, -1.0)
    ang = math.acos(dot_prod)
    return ang

def precision_recall_perpen(ang_diffs, neither_ang_diffs):
    '''
    Get the recall curve for the perpendicular case.
    Threshold on (ang_diffs - math.pi / 2.0), i.e., distance to 90 degree
    Input
        ang_diffs: a list of angular difference measured in radian, supposed to be perpendicular
        neither_ang_diffs: a list of angular difference measured in radian, supposed to be not perpendicular
    Output:
        The AUC score for the recall curve.
        And a plot of the curve stored in "perpen.pdf"
    '''
    plt.clf()
    n_pos_sample = len(ang_diffs)
    n_neg_sample = len(neither_ang_diffs)
    # This will be either TP or FN
    np_ang_diffs = np.array(ang_diffs).copy()
    np_ang_diffs = [val for val in np.abs(np_ang_diffs - (math.pi / 2.0))] # dist to 90 degree

    # This will be either FP or TN
    np_neither_ang_diffs = np.array(neither_ang_diffs).copy()
    np_neither_ang_diffs = [val for val in np.abs(neither_ang_diffs - (math.pi / 2.0))] # dist to 90 degree

    # recall = [0, 0] # Needs at least two points to calculate area under the curve
    # precision = [1, 1]

    all_ang_diffs = np_ang_diffs + np_neither_ang_diffs
    all_ang_diffs = convert_ang_to_prob(all_ang_diffs)
    true_labels = [1 for i in range(n_pos_sample)] + [0 for i in range(n_neg_sample)]
    precision, recall, _= precision_recall_curve(true_labels, all_ang_diffs)
    AP = average_precision_score(true_labels, all_ang_diffs)

    # snapped_uniq_angs = np.linspace(0, snap_thresh, num=100)
    # for ang in snapped_uniq_angs:
    #     tp = np.sum(np_ang_diffs <= ang)
    #     fp = np.sum(np_neither_ang_diffs <= ang)
    #     fn = np.sum(np_ang_diffs > ang)
    #     assert(tp + fn == n_pos_sample)
    #     assert(float(tp + fn) != 0)

    #     if not tp + fp == 0: # skip if recall is not calculatable
    #         recall.append(tp / float(tp + fn + 1e-6))
    #         precision.append(tp / float(tp + fp + 1e-6))

    # # REVIEW
    # # Remove zigzap patterns
    # # (Following Pascal VOC AP: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
    # recall, precision = np.array(recall), np.array(precision)
    # idx = np.argsort(recall)
    # recall = recall[idx]
    # precision = precision[idx]
    # # for i in range(len(precision)-1):
    # #     precision[i] = max(precision[i], np.max(precision[i+1:]))

    # AP = auc(x=recall, y=precision)

    plt.plot(recall, precision, 'b*-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.title('Perpendicular Precision-Recall Curve: AP = %.2f' % (AP) )
    plt.savefig("Perpendicular.pdf", bbox_inches='tight')

    return AP

def precision_recall_parall(ang_diffs, neither_ang_diffs):
    '''
    Get the recall curve for the parallel case.
    Threshold on the ang_diffs.
    Input
        ang_diffs: a list of angular difference measured in radian, supposed to be parallel
        neither_ang_diffs: a list of angular difference measured in radian, suppose to be not parallel
    Output:
        The AUC score for the recall curve, i.e. AP
        And a plot of the curve stored in "parall.pdf"
    '''
    plt.clf()
    n_pos_sample = len(ang_diffs)
    n_neg_sample = len(neither_ang_diffs)

    # This will be either TP or FN
    np_ang_diffs_ali = np.array(ang_diffs).copy() # when two vectors point to the same direction
    np_ang_diffs_rev = np.pi - np.array(ang_diffs) # when two vectors point to the opposite direction
    np_ang_diffs = [min(np_ang_diffs_ali[i], np_ang_diffs_rev[i]) for i in range(len(np_ang_diffs_rev))]

    # This will be either FP or TN
    np_neither_ang_diffs_ali = np.array(neither_ang_diffs).copy() # when two vectors point to the same direction
    np_neither_ang_diffs_rev = np.pi - np.array(neither_ang_diffs) # when two vectors point to the opposite direction
    np_neither_ang_diffs = [min(np_neither_ang_diffs_ali[i], np_neither_ang_diffs_rev[i]) for i in range(len(np_neither_ang_diffs_rev))]

    # recall = [0, 0]  # Needs at least two points to calculate area under the curve
    # precision = [1, 1]

    all_ang_diffs = np_ang_diffs + np_neither_ang_diffs
    all_ang_diffs = convert_ang_to_prob(all_ang_diffs)
    true_labels = [1 for i in range(n_pos_sample)] + [0 for i in range(n_neg_sample)]
    precision, recall, _= precision_recall_curve(true_labels, all_ang_diffs)
    AP = average_precision_score(true_labels, all_ang_diffs)

    # print('Average precision-recall score: {0:0.2f}'.format(AP))

    # snapped_uniq_angs = np.linspace(0, snap_thresh, num=100)
    # for ang in snapped_uniq_angs:
    #     tp = np.sum(np_ang_diffs <= ang)
    #     fp = np.sum(np_neither_ang_diffs <= ang)
    #     fn = np.sum(np_ang_diffs > ang)
    #     assert(tp + fn == n_pos_sample)
    #     assert(float(tp + fn) != 0)

    #     if not tp + fp == 0: # skip if recall is not calculatable
    #         recall.append(tp / float(tp + fn + 1e-6))
    #         precision.append(tp / float(tp + fp + 1e-6))

    # # REVIEW
    # # Remove zigzap patterns
    # # (Following Pascal VOC AP: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
    # recall, precision = np.array(recall), np.array(precision)
    # idx = np.argsort(recall)
    # recall = recall[idx]
    # precision = precision[idx]
    # # for i in range(len(precision)-1):
    # #     precision[i] = max(precision[i], np.max(precision[i+1:]))

    # AP = auc(x=recall, y=precision)  # TODO this might not be correct

    plt.plot(recall, precision, 'b*-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.title('Parallel Precision-Recall Curve: AP= %.2f' % (AP) )
    plt.savefig("Parallel.pdf", bbox_inches='tight')

    return AP

# Angles are in range from 0 to pi radians. Note that the perpendicular function already subtracts by pi/2 radians so we don't have to
# To compute precision recall, you need probabilities.
# Error of pi radians -> prob of 0.
# Error of 0 radians -> prob of 1.
def convert_ang_to_prob(angles):
  for i in range(len(angles)):
    angles[i] = 1.0 - (angles[i] / math.pi)
  return angles


def get_ang_diffs(pred_normal, xyxyr_tups):
    assert(len(pred_normal.shape) == 3)
    assert(pred_normal.shape[2] == 3)

    NEITHER_REL = "neither" # REVIEW

    unit_pred_normal = normalize(pred_normal)
    ang_diffs = []
    for xyxyr in xyxyr_tups:
        x_A = xyxyr[0]
        y_A = xyxyr[1]
        x_B = xyxyr[2]
        y_B = xyxyr[3]
        rel = xyxyr[4]
        assert(rel == "parallel" or rel == "perpendicular" or rel == NEITHER_REL)

        norm_A = unit_pred_normal[y_A, x_A, :]
        norm_B = unit_pred_normal[y_B, x_B, :]

        ang_diffs.append(ang_diff(norm_A, norm_B))

    perpendi_idx = [i for i in range(len(xyxyr_tups)) if xyxyr_tups[i][4] == 'perpendicular']
    parallel_idx = [i for i in range(len(xyxyr_tups)) if xyxyr_tups[i][4] == 'parallel']
    neither_idx = [i for i in range(len(xyxyr_tups)) if xyxyr_tups[i][4] == NEITHER_REL]
    
    
    perpendi_ang_diffs = [ang_diffs[i] for i in perpendi_idx]
    parallel_ang_diffs = [ang_diffs[i] for i in parallel_idx]
    neither_ang_diffs  = [ang_diffs[i] for i in neither_idx]

    return perpendi_ang_diffs, parallel_ang_diffs, neither_ang_diffs


def eval_rel_normal_by_ang(perpendi_ang_diffs, parallel_ang_diffs, neither_ang_diffs):
    '''
    Evaluate the relative normals of a list of predicted normal maps.
    Input:
        perpendi_ang_diffs 
        parallel_ang_diffs
        neither_ang_diffs
    Output:
        Two curves: x axis is the threshold, y axis is the recall.
        The area under the curve for the perpendicular and parallel curves.
    '''

    perpendi_ang_diffs, parallel_ang_diffs, neither_ang_diffs = np.array(perpendi_ang_diffs), np.array(parallel_ang_diffs), np.array(neither_ang_diffs)
    perpendi_ang_diffs = perpendi_ang_diffs[~np.isnan(perpendi_ang_diffs)]
    parallel_ang_diffs = parallel_ang_diffs[~np.isnan(parallel_ang_diffs)]
    neither_ang_diffs = neither_ang_diffs[~np.isnan(neither_ang_diffs)]

    perpendi_AP = precision_recall_perpen(perpendi_ang_diffs, neither_ang_diffs)
    parallel_AP = precision_recall_parall(parallel_ang_diffs, neither_ang_diffs)

    return perpendi_AP, parallel_AP


def eval_rel_normal(list_pred_normal, list_xyxyr_tups):
    '''
    Evaluate the relative normals of a list of predicted normal maps.
    Input:
        list_pred_normal: a list of numpy arrays HxWx3. The  predicted normal maps .
                          Does not assumed to be unit-normalized
                    [
                        normal map1,
                        normal map2,
                        ...
                    ]
        list_xyxyr_tups : a list of list of tuples (x_A, y_A, x_B, y_B, relation)
                    relation can be either "parallel", "perpendicular" or "neither"
                    [
                        [(x_A, y_A, x_B, y_B, relation), (x_A, y_A, x_B, y_B, relation), ...],  # for 1st normal map
                        [(x_A, y_A, x_B, y_B, relation), (x_A, y_A, x_B, y_B, relation), ...],  # for 2nd normal map
                        ...
                    ]

        snap_thresh_parall: beyond this thresh is no longer considered as parallel
        snap_thresh_perpen: beyond this thresh (i.e., distance to 90 degree) is no longer considered as perpendicular

    Output:
        Two curves: x axis is the threshold, y axis is the recall.
        The area under the curve for the perpendicular and parallel curves.
    '''

    assert(len(list_pred_normal) == len(list_xyxyr_tups))

    perpendi_ang_diffs = []
    parallel_ang_diffs = []
    neither_ang_diffs = []
    for pred_normal, xyxyr_tups in zip(list_pred_normal, list_xyxyr_tups):
        _perpendi, _parallel, _neither = get_ang_diffs(pred_normal, xyxyr_tups)
        perpendi_ang_diffs += _perpendi
        parallel_ang_diffs += _parallel
        neither_ang_diffs  += _neither


    perpendi_AP, parallel_AP = eval_rel_normal_by_ang(perpendi_ang_diffs, parallel_ang_diffs, neither_ang_diffs)

    return perpendi_AP, parallel_AP



