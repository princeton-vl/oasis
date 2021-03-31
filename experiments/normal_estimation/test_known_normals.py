import argparse
import os
import torch
import numpy as np
import cv2
import math
import sys

# sys.path.append('../common')
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import valid
from common.utils import save_obj, load_obj
# from common.datasets.SNOWDataset import SNOWDataset
from common.datasets.OASISDataset import OASISNormalDatasetVal
from common.models.NIPSNetwork import NIPSSurfaceNetwork


sys.path.append('../')
from torch.utils import data


def valid_normals(data_loader, max_iter, verbal, pred_folder, b_vis_normal = False):
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

  iter = 0
  with torch.no_grad():
    errors = []
    for step, data_tuple in enumerate(data_loader):

      inputs, targets, masks, target_res, img_name = data_tuple
      iter += 1

      if iter > max_iter:
        break
    
      
      try:
        pred_path = os.path.join(pred_folder, os.path.basename(img_name[0]).replace(".png", ".npy"))
        pred_np = np.load(pred_path)			# (H, W, 3)
      except Exception as e:
        return {"run_log": "Error loading %s" % os.path.basename(img_name[0]).replace(".png", ".npy")}

      try:
        orig_height = target_res[0]
        orig_width = target_res[1]
        # pred_np = cv2.resize(pred_np, (target_res[1], target_res[0]))
        pred_np = np.transpose(pred_np, (2, 0, 1))   # HWC to CHW.
        pred_np = pred_np[np.newaxis, :, :]
        output_var = torch.from_numpy(pred_np)
        output_var = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='bilinear')
        output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
        mask = masks.byte().squeeze(1) # remove color channel
        output = output_var.permute(0,2,3,1)[mask, :]
        target = targets.permute(0,2,3,1)[mask, :]
      except Exception as e:
        return {"run_log": str(e)}
     
      error = angle_error(output, target)
      errors.append(error.data.cpu().detach().numpy())

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

    results = {"run_log": "success"}
    results['MAE'] = MAE
    results['MDAE'] = MDAE
    results['11.25'] = below_1125
    results['22.5'] = below_225
    results['30'] = below_30
    return results


# python test_known_normals.py -p ./nom_OASIS_0.0001 -t /home/wfchen/OASIS_trainval/csv_after_clean/OASIS_test.csv
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_file', '-t', default='NYU_test_50_50_1000.csv')   # or DIW_test.csv 
  parser.add_argument('--pred_folder', '-p', default=None)
  parser.add_argument('--num_iters', '-iter', default=100000, type=int)
  parser.add_argument('--output_file', '-o', default=None)
  parser.add_argument('--vis_normal', '-vis', action='store_true', default=False)
  
  
  args = parser.parse_args()

  collate_fn = default_collate
  DataSet = OASISNormalDatasetVal
  test_dataset = DataSet(csv_filename = args.test_file)
  test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = collate_fn)
  print("Testing on %s" % args.test_file)


  normal_result = valid_normals(data_loader = test_data_loader, max_iter = args.num_iters, verbal=True, 
                       pred_folder = args.pred_folder, b_vis_normal = args.vis_normal)
  print(normal_result)

  if args.output_file is not None:
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_obj(normal_result, args.output_file)

