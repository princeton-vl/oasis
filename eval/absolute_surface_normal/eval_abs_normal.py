'''
  Written by Weifeng Chen
  Reviewed by Noriyuki Kojima 04/10/2019, 04/29/2019
  Modified by Weifeng Chen 04/28/2019
  Modified by David Fan 7/27/2019
  Reviewed by Weifeng Chen 03/08/2020
'''
import numpy as np
import math

def ang_error(pred_normal, gt_normal, ROI = None):
	'''
	Inputs
		pred_normal: A numpy array of HxWx3, float32
		gt_normal:   A numpy array of HxWx3, float32
		ROI:		 A numpy array of HxW,   uint8.
					 If None, the entire image is ROI
	Return
		The angular differences between pred and gt in the ROI
	'''

	assert(pred_normal.shape[0] == gt_normal.shape[0])
	assert(gt_normal.shape[0] == gt_normal.shape[0])
	assert(len(pred_normal.shape) == 3 and len(gt_normal.shape) == 3)

	# normalize
	pred_normal = pred_normal / np.linalg.norm(pred_normal, ord=2, axis=2, keepdims=True)	# HxWx3
	gt_normal = gt_normal / np.linalg.norm(gt_normal, ord=2, axis=2, keepdims=True)		# HxWx3

	# calculate the angle difference
	dot_prod = np.multiply(pred_normal, gt_normal)				# HxWx3
	dot_prod = np.sum(dot_prod, axis = 2)						# HxW
	dot_prod = np.clip(dot_prod, a_min = -1.0, a_max = 1.0)

	angles = np.arccos(dot_prod)
	angles = np.degrees(angles)
	

	if ROI is None:
		assert(np.all(angles <= 180.))
		return angles.flatten()
	else:
		assert(np.all(angles[ROI > 0] <= 180.))
		return angles[ROI > 0]


def evaluate_normal(pred_normals, gt_normals, ROIs = None):
	'''
	Inputs
		pred_normal: A list of numpy arrays of HxWx3
		gt_normal:   A list of numpy arrays of HxWx3
		ROIs:		 A list of numpy arrays of HxW.
	Return
		mean_err:    The mean angular difference between the predicted and ground-truth normals. Measured in degree.
		median_err:  The median angular difference between the predicted and ground-truth normals. Measured in degree.
		below_11_25: The percentage of pixels whose angular difference is less than 11.25 degree.
		below_22_5:  The percentage of pixels whose angular difference is less than 22.50 degree.
		below_30:    The percentage of pixels whose angular difference is less than 30.00 degree.
	'''

	assert(len(pred_normals) == len(gt_normals))

	if ROIs is None:
		ROIs = [None for i in range(len(gt_normals))]

	angles = []
	for pred, gt, ROI in zip(pred_normals, gt_normals, ROIs):
		_angles = ang_error(pred, gt, ROI)
		angles.append(_angles)

	angles = np.concatenate(angles)
	n_pixels = float(len(angles))

	mean_err = np.mean(angles)
	median_err = np.median(angles)
	below_11_25 = float(np.sum(angles < 11.25)) / n_pixels * 100
	below_22_5  = float(np.sum(angles < 22.5))  / n_pixels * 100
	below_30    = float(np.sum(angles < 30))    / n_pixels * 100

	return mean_err, median_err, below_11_25, below_22_5, below_30
