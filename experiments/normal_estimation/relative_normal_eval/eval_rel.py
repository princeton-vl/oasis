import argparse
import h5py
import os
import torch
import math
import numpy as np
import cv2
import pickle
import sys
from collections import defaultdict
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(cwd)


# metric
sys.path.append(cwd + '/../../eval/relative_surface_normal')
from eval_rel_normal import get_ang_diffs, eval_rel_normal_by_ang

from common.utils import save_obj, load_obj, makedir_if_not_exist
from common.models.NIPSNetwork import NIPSSurfaceNetwork

from torch.utils import data
from torch.utils.data.dataloader import default_collate

def is_data_parallel(state_dict):
  for key in state_dict:
    if 'module' in key:
      return True
  return False

def get_coord_change(in_coord_sys, out_coord_sys):
  coord_change = [1.,1.,1.]
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
  return coord_change

# normalize normals
def normalize(normal_map):
  l2_norms = np.linalg.norm(normal_map, axis = 2)
  l2_norms = np.stack([l2_norms, l2_norms, l2_norms], axis = -1)
  return np.divide(normal_map, l2_norms)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_file', '-t')  
  parser.add_argument('--model_file', '-model', default=None)
  parser.add_argument('--in_coord_sys', '-in_coord_sys', type=str, required=True)
  parser.add_argument('--out_coord_sys', '-out_coord_sys', type=str, required=True)
  parser.add_argument('--exp_name', '-exp_name', type=str, required=True)
  parser.add_argument('--output_file', '-o', type=str, required=True)
  parser.add_argument('--zhang_preds_path',  type=str, required=False)
  parser.add_argument('--front_facing', '-front_facing', action='store_true', default=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  args = parser.parse_args()
  args = vars(args)
  in_coord_sys = args['in_coord_sys']
  out_coord_sys = args['out_coord_sys']
  exp_name = args['exp_name']

  assert(in_coord_sys in ['OASIS', 'NYU', 'SNOW'])
  assert(out_coord_sys in ['OASIS', 'NYU', 'SNOW'])

  if args['model_file'].find(".bin") >= 0:
    mode = 'model'
    training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args['model_file'])), 'args.pkl'))

    print("#######################################################################")
    print("Testing a model, args: {}".format(args))
    print("#######################################################################")
    
    NetworkType = {"NIPSSurface": NIPSSurfaceNetwork}  
    
    model = NetworkType[training_args['model_name']]().to(device)
    model_name = training_args['model_name']

    state_dict = torch.load(args['model_file'])
    is_data_parallel = is_data_parallel(state_dict)
    print('Using DataParallel: {}'.format(is_data_parallel))
    if is_data_parallel:
      model = torch.nn.DataParallel(model)
      model.load_state_dict(state_dict)  
      model = model.module
    else:
      model.load_state_dict(state_dict) 
    
    model.eval()
    
  elif args['model_file'] == "Zhang":
    print("#######################################################################")
    print("Testing Zhang:")
    print("#######################################################################")
    mode = 'Zhang'
  else:
    print("#######################################################################")
    print("Testing on pre-estimated normals:", args['model_file'])
    print("#######################################################################")
    mode = 'pre-estimated'
  
  # Assemble tuples. Note that order in file is randomized.
  test_file = args['test_file']
  
  perpendi_ang_diffs = []
  parallel_ang_diffs = []
  neither_ang_diffs = []

  image_pairs_map = defaultdict(list)
  for line in open(test_file, 'r'):
    im_name, y_A, x_A, y_B, x_B, rel = line.strip().split(',')
    x_A = int(x_A)
    y_A = int(y_A)
    x_B = int(x_B)
    y_B = int(y_B)

    image_pairs_map[im_name].append((x_A, y_A, x_B, y_B, rel))

  # Predict normals
  for im_name in image_pairs_map:
    color = cv2.imread(im_name).astype(np.float32)
    anno_id = im_name.split('/')[-1].replace('.png', '')
    orig_height, orig_width = color.shape[:2]
    color = cv2.resize(color, (320, 240))
    color = np.transpose(color, (2, 0, 1)) / 255.0
    color = torch.tensor(color).to(device).unsqueeze(0)
    
    print(im_name, "front facing = %s" % args["front_facing"])

    # Do inference
    if mode == "Zhang":
      pred_filename = '%s/Surface_%s_normal_est.h5' % (args['zhang_preds_path'], anno_id)
      print("Zhang:", pred_filename)
      f = h5py.File(pred_filename, 'r')
      pred_normal = np.array(f['normals_float']) # 3 x 240 x 320
      pred_normal = np.transpose(pred_normal, (1,2,0)) # 240 x 320 x 3
      f.close()

      # Resize normal pred to original resolution
      output = cv2.resize(pred_normal, (orig_width, orig_height),interpolation=cv2.INTER_LINEAR)
      # Fix color channels
      output = output[:,:,[0,2,1]] # Flip their y and z axes
      output[:,:,1] *= -1.
      output[:,:,2] *= -1.

    elif mode == 'model':
      output = model(color)
      # Resize to groundtruth resolution
      output = torch.nn.functional.interpolate(output, (orig_height, orig_width), mode='bilinear')
      output = output.cpu().detach().numpy()
      coord_change = get_coord_change(in_coord_sys, out_coord_sys)
      output[:,0,:,:] *= coord_change[0]
      output[:,1,:,:] *= coord_change[1]
      output[:,2,:,:] *= coord_change[2]
      # Model does inference on 4D tensor, but eval function takes 3D
      # Has to be HxWx3
      output = np.transpose(output[0,:,:,:], (1,2,0))

    elif mode == 'pre-estimated':
      try:
        pred_path = os.path.join(args['model_file'], anno_id + '.npy')
        print("pre-estimated:", pred_path)
        pred_normal = np.load(pred_path)
        output = cv2.resize(pred_normal, (orig_width, orig_height)) # HxWx3
      except Exception as e:
        print("Error:", str(e))
        save_obj({"run_log": "Error processing %s.npy" % anno_id}, args['output_file'])
        import sys
        sys.exit()
        


    # For confirming the ground truth leads to AP 1.0 :)
    #normal_name = '/n/fs/pvl/datasets/3SIW/normal/{}.pkl'.format(anno_id)
    #dict = pickle.load(open(normal_name, 'rb'))
    #normal = np.zeros((orig_height, orig_width, 3))
    #normal[dict['min_x']:dict['max_x']+1, dict['min_y']:dict['max_y']+1, :] = dict['normal']
    #output = normal
  
    if args['front_facing']:
      output[:,:,0] = 0
      output[:,:,1] = 0
      output[:,:,2] = 1

    _perpendi, _parallel, _neither = get_ang_diffs(pred_normal = output, xyxyr_tups = image_pairs_map[im_name])
    perpendi_ang_diffs += _perpendi
    parallel_ang_diffs += _parallel
    neither_ang_diffs  += _neither

  print('Evaluating:')
  perpendi_AP, parallel_AP = eval_rel_normal_by_ang(perpendi_ang_diffs, parallel_ang_diffs, neither_ang_diffs)  
  print('Perpendicular AP: {}, Parallel AP: {}'.format(perpendi_AP, parallel_AP))

  save_obj({'perpendi_AP':perpendi_AP, 'parallel_AP':parallel_AP, "run_log": "success"}, args['output_file'])
