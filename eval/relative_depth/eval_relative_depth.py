'''
  Written by Weifeng Chen
  Reviewed by Noriyuki Kojima 04/10/2019
'''
import numpy as np
import random

def eval_relative_depth(pred_depth, gts):
	######################################################
	# Inputs
	#	pred_normal: A numpy array of HxW
	#   gts: An array of list / numpy array [x_A, y_A, x_B, y_B, relation]
	#					relation =  1 if depth_A > depth_B,
	#					relation = -1 if depth_A <= depth_B
	# Return
	#   n_correct:   Number of pairs that are correct
	#   n_total:     total number of pairs
	######################################################
	assert(len(pred_depth.shape) == 2)
	assert(np.all(pred_depth >= 0))

	n_point = len(gts)
	n_correct = 0

	for elem in gts:
		depth_A = pred_depth[elem[1], elem[0]]
		depth_B = pred_depth[elem[3], elem[2]]

		if depth_A > depth_B:
			_classify_res = 1
		else:
			_classify_res = -1
		if _classify_res == elem[4]:
			n_correct += 1
	return n_correct, n_point
