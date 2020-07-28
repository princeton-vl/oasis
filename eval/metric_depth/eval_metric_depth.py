'''
  Written by David Fan
  Reviewed by Weifeng Chen: 04/20/2019
'''
import numpy as np
import sys

######################################################
# Inputs
# output: A numpy array of HxW. Values correspond to metric depth estimate.
# ground_truth: A numpy array of HxW. Values correspond to metric depth.
# Return
#   RMSE     : linear root mean square error
#   RMSE_log : logarithmic root mean square error
#   RMSE_inv : logarithmic (scale-invariant) root mean square error
#   ABS_REL  : absolute relative difference
#   SQR_REL  : squared relative difference
######################################################
def evaluate_metric_depth(output, ground_truth):
  mask = ground_truth > 0
  output = np.copy(output[mask])
  ground_truth = np.copy(ground_truth[mask])
  # All metrics defined in section 4.3 of this paper; https://arxiv.org/abs/1406.2283v1
  # Calculate RMSE
  RMSE = np.sqrt(np.mean((output - ground_truth)**2))
  # Calculate RMSE_log (with natural log)
  transformed = np.log(output)
  transformed_truth = np.log(ground_truth)
  RMSE_log = np.sqrt(np.mean((transformed - transformed_truth)**2))
  # Calculate RMSE_inv (scale invariant log)
  alpha = np.mean(transformed_truth - transformed)
  RMSE_inv = np.sqrt(np.mean((transformed - transformed_truth + alpha) ** 2))
  # Calculate ABS_REL
  ABS_REL = np.mean(np.abs(output - ground_truth) / ground_truth)
  # Calculate SQR_REL
  SQR_REL = np.mean((output - ground_truth)**2 / ground_truth)

  return RMSE, RMSE_log, RMSE_inv, ABS_REL, SQR_REL


######################################################
# Utility function
######################################################
def _back_project(depth, focal_x, focal_y, mask):
  '''
    return Nx3 XYZ
  '''
  height, width  = depth.shape[:2]
  
  xs = np.linspace(0, width-1, width)
  ys = np.linspace(0, height-1, height)
  xv, yv = np.meshgrid(xs, ys)
  
  xv -= 0.5 * width
  yv -= 0.5 * height

  X = xv / (focal_x + 1e-8) * depth
  Y = yv / (focal_y + 1e-8) * depth
  Z = depth

  X = X[mask]
  Y = Y[mask]
  Z = Z[mask]
  
  XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()]).transpose()   # N by 3

  return XYZ

######################################################
# Utility function
######################################################
def _least_scale_trans(ref, query):
    a = np.sum(np.multiply(ref, query))
    b = np.sum(np.square(query))
    scale = a / b

    return scale

######################################################
# Utility function
######################################################
def _vis_point_cloud(filename, XYZ, color):
  if len(color) == 3:
    with open(filename, "w") as f:
      for i in range(XYZ.shape[0]):
        f.write("v %g %g %g %d %d %d\n" % (XYZ[i,0], XYZ[i,1], XYZ[i,2], color[0], color[1], color[2]))
  else:
    max_val = np.max(color)
    with open(filename, "w") as f:
      for i in range(XYZ.shape[0]):
        f.write("v %g %g %g %g %g %g\n" % (XYZ[i,0], XYZ[i,1], XYZ[i,2], color[i] / max_val, color[i]/ max_val, color[i]/ max_val))



#######################################################################################################
# IMPORTANT: this function calculates the least euc distance between
#           (X_pred, Y_pred, Z_pred) and [scale * (X_gt, Y_gt, Z_gt) + (0,0,lambda)]  
#           Noted we are transforming gt, not pred!
# Input:
#   pred:   Nx3 XYZ
#   gt:     Nx3 XYZ
#######################################################################################################
def _scale_XYZ_translate_Z(gt, pred, debug = False):

    assert(pred.shape[1] == 3)
    assert(gt.shape[0] == pred.shape[0])

    N_pt = gt.shape[0] + 1e-8

    x1 = gt[:,0]
    y1 = gt[:,1]
    z1 = gt[:,2]

    x2 = pred[:,0]
    y2 = pred[:,1]
    z2 = pred[:,2]

    X1_2 = np.sum(np.square(x1))
    Y1_2 = np.sum(np.square(y1))
    Z1_2 = np.sum(np.square(z1))
    X1X2 = np.sum(np.multiply(x1, x2))
    Y1Y2 = np.sum(np.multiply(y1, y2))
    Z1Z2 = np.sum(np.multiply(z1, z2))

    Z1_sum = np.sum(z1)
    Z2_sum = np.sum(z2)

    # z2x1_2 = np.sum(np.multiply(z2, np.square(x1)))
    # z2y1_2 = np.sum(np.multiply(z2, np.square(y1)))
    # z1x1x2 = np.sum(np.multiply(z1, np.multiply(x1, x2)))
    # z1y1y2 = np.sum(np.multiply(z1, np.multiply(y1, y2)))

    # denominator = X1_2 + Y1_2 + 1e-8
    # scale = (X1X2 + Y1Y2) / denominator
    # delta = (z2x1_2 + z2y1_2 - z1x1x2 - z1y1y2) / denominator
    
    denominator = X1_2 + Y1_2 + Z1_2 - Z1_sum * Z1_sum / N_pt
    scale2 = (X1X2 + Y1Y2 + Z1Z2 - Z1_sum * Z2_sum / N_pt) / denominator
    delta2 = (Z2_sum - scale2 * Z1_sum) / N_pt


    # loss = np.sum(np.power(np.matmul(A, [[scale], [delta]]) - b,2))

    # temp = scale * gt     
    # temp[:,-1] += delta    
    # loss = np.sum(np.square( temp - pred ))
    # print(loss, scale, delta)

    temp = scale2 * gt     
    temp[:,-1] += delta2 

    euc_dists = np.sqrt(np.sum(np.square( temp - pred ), axis = 1))


    if debug:
      _vis_point_cloud("transformed.obj", temp, euc_dists)


    return euc_dists, {1:scale2, 'translation':delta2}


######################################################
# Least euc dist under scaling in XYZ and translation along Z axis
# Inputs
# pred: A numpy array of HxW. Values correspond to metric depth estimate.
# gt: A numpy array of HxW. Values correspond to metric depth.
# focal_pred: Predicted focal length, a float
# focal_gt: Predicted focal length, a float
# Return
#   euc_dists: the euc distance between each point pair
######################################################
def point_cloud_euclidean(pred, gt, focal_pred, focal_gtx, focal_gty, debug):
    assert(pred.shape[0] == gt.shape[0] and pred.shape[1] == gt.shape[1])

    gt = np.copy(gt)
    pred = np.copy(pred)   

    ################################################################
    # back project and normalize the predicted depth
    mask = gt > 0
    XYZ_pred = _back_project(pred, focal_pred, focal_pred, mask)
    XYZ_gt   = _back_project(gt, focal_gtx, focal_gty, mask)

    euc_dists, transformation = _scale_XYZ_translate_Z(pred = XYZ_gt, gt = XYZ_pred, debug = debug)  # pay attention to the order!
       
    # scale = _least_scale_trans(ref = XYZ_gt, query = XYZ_pred)
    # euc_dists = np.sqrt(np.sum(np.square(scale * XYZ_pred - XYZ_gt), axis = 1))

    # debug
    if debug:
      _vis_point_cloud("XYZ_gt.obj", XYZ_gt, [255,0,0])
      _vis_point_cloud("XYZ_pred.obj", XYZ_pred, [128,128,128])
      input("=================================")

    return euc_dists, transformation
    

######################################################
# unit test
######################################################
def _unit_test():
    pred = np.random.rand(10, 3)
    # pred = np.ones((10, 3))
    gt = 5.123 * pred 
    gt[:, -1] += 10.0

    dist, trans = _scale_XYZ_translate_Z(gt = pred, pred = gt)
    print(trans)
