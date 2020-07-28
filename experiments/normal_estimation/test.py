import argparse
import os
import torch
import numpy as np
import cv2
import sys
sys.path.append('../common')

import valid
from common.utils import save_obj, load_obj
from common.datasets.SNOWDataset import SNOWDataset
from common.datasets.OASISDataset import OASISNormalDatasetVal
from common.models.NIPSNetwork import NIPSSurfaceNetwork

from torch.utils import data
from torch.utils.data.dataloader import default_collate


def is_data_parallel(state_dict):
  for key in state_dict:
    if 'module' in key:
      return True
  return False

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_file', '-t', default='NYU_test_50_50_1000.csv')   # or DIW_test.csv 
  parser.add_argument('--valid_file', '-v', default=None)
  parser.add_argument('--num_iters', '-iter', default=100000, type=int)
  parser.add_argument('--model_file', '-model', default=None)
  parser.add_argument('--output_file', '-o', default=None)
  parser.add_argument('--vis_normal', '-vis', action='store_true', default=False)
  parser.add_argument('--in_coord_sys', '-in_coord_sys', type=str, required=True)
  parser.add_argument('--out_coord_sys', '-out_coord_sys', type=str, required=True)
  parser.add_argument('--front_facing', '-front_facing', action='store_true', default=False)
  parser.add_argument('--best', '-best', action='store_true', default=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  args = parser.parse_args()
  training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args.model_file)), 'args.pkl'))

  print("#######################################################################")
  print("Testing args: {}".format(args))
  print("Training args: {}".format(training_args))
  print("#######################################################################\n\n\n")
  
  NetworkType = {"NIPSSurface": NIPSSurfaceNetwork}  
  
  model = NetworkType[training_args['model_name']]().to(device)

  model_name = training_args['model_name']
  in_coord_sys = args.in_coord_sys
  out_coord_sys = args.out_coord_sys
  if not args.front_facing:
    state_dict = torch.load(args.model_file)
    is_data_parallel = is_data_parallel(state_dict)
    print('Using DataParallel: {}'.format(is_data_parallel))
    if is_data_parallel:
      model = torch.nn.DataParallel(model)
      model.load_state_dict(state_dict)  
      model = model.module
    else:
      model.load_state_dict(state_dict)
  
  in_thresh = 0.0
  model.eval()

  if model_name == 'NIPSSurface' and 'SNOW' in args.test_file:
    collate_fn = default_collate
    DataSet = SNOWDataset
    dataset_name = 'SNOWDataset'
    test_dataset = DataSet(csv_filename=args.test_file)

    
  elif 'OASIS' in args.test_file:
    collate_fn = default_collate
    DataSet = OASISNormalDatasetVal
    dataset_name = 'OASISNormalDatasetVal'
    test_dataset = DataSet(csv_filename = args.test_file)
    

  test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = collate_fn)
  print("Testing on %s" % args.test_file)
  test_rel_error = valid.valid(model, in_coord_sys, out_coord_sys, test_data_loader, dataset_name, 
                               max_iter = args.num_iters, in_thresh=in_thresh, b_vis_normal=args.vis_normal, 
                               verbal=True, front_facing=args.front_facing)
  model.train()
  print(test_rel_error)

  if args.output_file is not None:
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_obj({'test_rel_error':test_rel_error}, args.output_file)

