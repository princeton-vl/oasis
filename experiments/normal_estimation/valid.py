import argparse
import os
import torch
import numpy as np
import cv2
import math
import sys

sys.path.append('../')
from torch.utils import data
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rot_normal(yaw, pitch, roll, normal):
  '''
  Input:
    normal: Nx3 numpy tensor
  Output:
    Nx3 numpy tensor
  '''
  
  mat = np.copy(normal).transpose()
  # mat = np.stack([in_nor[:,:,0].reshape(h*w), in_nor[:,:,1].reshape(h*w), in_nor[:,:,2].reshape(h*w)], 0)
  
  sin = math.sin(yaw)
  cos = math.cos(yaw)
  rotx = np.array([[1,0,0], [0, cos, -sin], [0, sin, cos]])
  mat = np.matmul(rotx, mat)  
  
  sin = math.sin(pitch)
  cos = math.cos(pitch)
  roty = np.array([[cos,0,sin], [0, 1, 0], [-sin, 0, cos]])
  mat = np.matmul(roty, mat)  

  sin = math.sin(roll)
  cos = math.cos(roll)
  rotz = np.array([[cos,-sin,0], [sin, cos, 0], [0, 0, 1]])
  mat = np.matmul(rotz, mat)  

  return mat.transpose()

def valid(model, in_coord_sys, out_coord_sys, data_loader, dataset_name, max_iter=1400, 
          verbal=False, b_vis_normal=False, in_thresh = None, front_facing = False):
  print("Evaluation on {}".format(dataset_name))
  coord_change = [1.,1.,1.]
  # NYU: x points left, y points down, z points toward us
  # SNOW: x points right, y points up, z points toward us
  # OASIS: x points right, y points down, z points toward us
  if in_coord_sys == 'NYU':
    if out_coord_sys == 'SNOW':
      coord_change = [-1.,-1.,1.]
    elif out_coord_sys == 'OASIS':
      coord_change = [-1.,1.,1.]
  elif in_coord_sys == 'SNOW':
    if out_coord_sys == 'NYU':
      coord_change = [-1.,-1.,1.]
    elif out_coord_sys == 'OASIS':
      coord_change = [1.,-1.,1.]
  elif in_coord_sys == 'OASIS':
    if out_coord_sys == 'NYU':
      coord_change = [-1.,1.,1.]
    elif out_coord_sys == 'SNOW':
      coord_change = [1.,-1.,1.]
  
  assert(in_coord_sys in ['SNOW', 'OASIS', 'NYU'])
  assert(out_coord_sys in ['SNOW', 'OASIS', 'NYU'])
  assert(dataset_name in ['SNOWDataset', 'NYUNormalDataset', 'OASISNormalDatasetVal'])

  if dataset_name == 'SNOWDataset':
    return valid_normals(model, 'SNOW', coord_change,  data_loader, max_iter, verbal, front_facing, b_vis_normal)
  elif dataset_name == 'OASISNormalDatasetVal':
    return valid_normals(model, 'OASIS', coord_change, data_loader, max_iter, verbal, front_facing, b_vis_normal)

def vis_normal(im, iter, is_color, is_groundtruth):
  im = im.transpose(1, 2, 0)
  if not is_color:
    # BGR -> RGB
    im = im[:,:,[2,1,0]]
  im = (im + 1) / 2 * 255.0
  #im = im - np.min(im)
  #im = im / np.max(im) * 255.0
  if is_groundtruth:
    cv2.imwrite('debug_img/{}_groundtruth.png'.format(iter), im)
  else:
    cv2.imwrite('debug_img/{}.png'.format(iter), im)

def valid_normals(model, type, coord_change, data_loader, max_iter, verbal, front_facing = False, b_vis_normal = False):
  def angle_error(preds, truths):
    '''
    preds and truths: Nx3 pytorch tensor
    '''
    preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
    truths_norm = torch.nn.functional.normalize(truths, p=2, dim=1)
    angles = torch.sum(preds_norm * truths_norm, dim=1)
    # Clip values so that max is 1 and min is -1, but don't change intermediate values
    angles = torch.clamp(angles, -1, 1)
    angles = torch.acos(angles)
   # mean_loss = torch.mean(loss)
    return angles

  # In degrees
  def mean(errors):
    error_sum = 0
    total_pixels = 0
    for matrix in errors:
      error_sum += np.sum(matrix)
      total_pixels += matrix.size
    return math.degrees(error_sum / total_pixels)

  # In degrees
  def median(errors):
    return math.degrees(np.median(np.concatenate(errors)))
  
  # 11.25, 22.5, 30
  def below_threshold(errors, thresh_angle):
    num = 0
    total_pixels = 0
    for matrix in errors:
      num += np.sum(matrix < math.radians(thresh_angle))
      total_pixels += matrix.size
    return num / total_pixels

  
  print("####################################")
  print("Evaluating...")
  print("\tfront_facing = %s"  %  front_facing)

  assert not model.training

  iter = 0
  with torch.no_grad():
    errors = []
    for step, data_tuple in enumerate(data_loader):
      if len(data_tuple) == 3:
        inputs, targets, target_res = data_tuple
      elif len(data_tuple) == 5:
        inputs, targets, masks, target_res, _ = data_tuple
        masks = masks.to(device)
      iter += 1
      if iter > max_iter:
        break
      input_var = Variable(inputs.to(device))
      output_var = model(input_var)
      targets = targets.to(device)

      # Shift coordinate systems
      output_var[:,0,:,:] *= coord_change[0]
      output_var[:,1,:,:] *= coord_change[1]
      output_var[:,2,:,:] *= coord_change[2]

      if type == 'SNOW':
        target_normal = targets[:,:3].to(device)
        coords = targets[:,3:].long() # must be long to index
        # Get normals using coordinates (one per image)
        pred_normals = torch.Tensor().to(device)
        # Might not have exactly batch size elements
        output = output_var#.clone().cpu().detach().numpy()
        for i in range(0, coords.shape[0]):
          # Coordinates in original resolution
          # Resize output to batchsize x 3 x orig_width x orig_height
          x = coords[i,0]
          y = coords[i,1]
          orig_height = target_res[0][i]
          orig_width = target_res[1][i]
          rescaled = torch.nn.functional.interpolate(output, (orig_height, orig_width), mode='bilinear')
          curr_normal = rescaled[i,:,x,y]
          pred_normals = torch.cat((pred_normals, curr_normal), -1)
        pred_normals = torch.reshape(pred_normals, (coords.shape[0], 3)).double().to(device)
        #output = torch.nn.functional.normalize(output, p=2, dim=1)
        #vis_normal(output[0,:,:,:].cpu().detach().numpy(), iter, step, False, False)
        #cv2.imwrite('debug_img/{}_{}_color.png'.format(iter, step), inputs[0,:,:,:].cpu().detach().numpy().transpose(1,2,0)*255)

        if front_facing:
          pred_normals[:,0] = 0 # pred_normals: torch.Size([1, 3])
          pred_normals[:,1] = 0
          pred_normals[:,2] = 1
        # print(pred_normals[:,0], pred_normals[:,1], pred_normals[:,2])
        error = angle_error(pred_normals, target_normal)

      elif type == 'OASIS':
        orig_height = target_res[0]
        orig_width = target_res[1]
        output_var = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='bilinear')
        targets = torch.nn.functional.interpolate(targets, size=(orig_height, orig_width), mode='bilinear')
        masks = torch.nn.functional.interpolate(masks, size=(orig_height, orig_width), mode='bilinear')
        mask = masks.byte().squeeze(1) # remove color channel
        output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
        output = output_var.permute(0,2,3,1)[mask, :]
        target = targets.permute(0,2,3,1)[mask, :]

        if front_facing :
          output[:,0] = 0 # output: torch.Size([N, 3])
          output[:,1] = 0
          output[:,2] = 1
        # print(torch.max(target[:,2]), torch.min(target[:,2]))


        
        error = angle_error(output, target)

        #print('{}_{}: {}'.format(iter, step, math.degrees(torch.mean(error))))
        
        #vis_normal(targets[0,:,:,:].cpu().detach().numpy(), iter, is_color=False, is_groundtruth=True)
        #cv2.imwrite('debug_img/{}_{}_mask.png'.format(iter, step), masks[0,:,:,:].cpu().detach().numpy().transpose(1,2,0)*255)

        # vis_normal(output_var[0,:,:,:].cpu().detach().numpy(), iter, is_color=False, is_groundtruth=False)
        # cv2.imwrite('debug_img/{}_color.png'.format(iter), inputs[0,:,:,:].cpu().detach().numpy().transpose(1,2,0)*255)

      errors.append(error.data.cpu().detach().numpy())

      if b_vis_normal:
        vis_normal(output_var[0,:,:,:].cpu().detach().numpy(), iter, is_color=False, is_groundtruth=False)

      if iter % 50 == 0:
        print("Iter {}: {} mean radian error".format(iter, np.mean(np.concatenate(errors))))
        sys.stdout.flush()
    MAE = mean(errors)
    print("Mean angle error: {} degs".format(MAE))
    below_1125 = below_threshold(errors, 11.25)
    print("% below 11.25 deg: {}".format(below_1125))
    below_225 = below_threshold(errors, 22.5)
    print("% below 22.5 deg: {}".format(below_225))
    below_30 = below_threshold(errors, 30)
    print("% below 30 deg: {}".format(below_30))
    MDAE = median(errors)
    print("Median angle error: {} degs".format(MDAE))
    sys.stdout.flush()

    results = {}
    results['MAE'] = MAE
    results['MDAE'] = MDAE
    results['11.25'] = below_1125
    results['22.5'] = below_225
    results['30'] = below_30
    return results
