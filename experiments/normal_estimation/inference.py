import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2
import math
import sys

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, cwd)
sys.path.insert(0, cwd + '/common')


from torch.utils import data
from torch.autograd import Variable
from common.utils import save_obj, load_obj, makedir_if_not_exist
from common.models.NIPSNetwork import NIPSSurfaceNetwork

from torch.utils import data
from torch.utils.data.dataloader import default_collate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vis_normal(im):
  im = np.copy(im.transpose(1, 2, 0))
  im = (im + 1) / 2 * 255.0

  return im
  
def run(model, list_of_imgs, out_folder, b_vis_normal=False):
  
  print("####################################")
  print("Infering... Saving to %s" % out_folder)

  assert not model.training

  iter = 0
  with torch.no_grad():
    for img_fname in tqdm(list_of_imgs):
      ################################################
      color = cv2.imread(img_fname).astype(np.float32)
      target_res = color.shape[:2]
      color = cv2.resize(color, (320, 240))
      color = color.transpose(2, 0, 1).astype(np.float32) / 255.0	

      color = np.expand_dims(color, axis = 0)
      inputs = torch.from_numpy(color)	

      input_var = Variable(inputs.to(device))
      output_var = model(input_var)
      # output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)

      pred_np = output_var.cpu().detach().numpy()[0,:,:,:]
      pred_np = pred_np.transpose(1, 2, 0)
      
      


      #######################################################################
      # save to output folder
      # orig_size_pred_np = cv2.resize(pred_np, (target_res[1], target_res[0]))
      img_id = os.path.basename(img_fname).replace(".png", "")      
      np.save("%s/%s.npy" % (out_folder, img_id), pred_np)
      # print("Done saving to %s/%s.npy" % (out_folder, img_id))

      if b_vis_normal:
        vis = (pred_np + 1.0) / 2.0 * 255.0
        cv2.imwrite("%s/%s.png" % (out_folder, img_id), vis.astype(np.uint8))


def get_img_list(csv_filename):
  img_names = []
  
  with open(csv_filename, "r") as f:
    lines = [line.strip() for line in f]
    lines = lines[1:] 	#skip the header
  
  '''
    Image,FocalLength,Mask,Normal,Depth,RelativeDepth,Fold,Occlusion,SharpOcc,SmoothOcc,SurfaceSemantic,PlanarInstance,SmoothInstance,ContinuousInstance
  '''
  for line in lines:
    splits = line.split(",")
    if splits[-1] != '' and splits[4] != '':
      img_names.append(splits[0])
    
  return img_names


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_csv', type=str, default='') 
  parser.add_argument('--model_file', '-model', default=None)
  parser.add_argument('--out_folder', '-out_folder', default=None)
  parser.add_argument('--vis_normal',  '-vis', action='store_true', default=False)

  args = parser.parse_args()
  training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args.model_file)), 'args.pkl'))
  print("#######################################################################")
  print("Testing args: {}".format(args))
  print("Training args: {}".format(training_args))
  print("#######################################################################\n\n\n")
  
      
  NetworkType = {"NIPSSurface": NIPSSurfaceNetwork}  
  
  ############ load trained model
  model = NetworkType[training_args['model_name']]().to(device)
  model_name = training_args['model_name']

  state_dict = torch.load(args.model_file)
  model.load_state_dict(state_dict)
  
  model.eval()


  ############ inference
  out_folder = args.out_folder
  if args.out_folder is None:
    out_folder = args.model_file.replace(".bin", "")
  os.makedirs(out_folder, exist_ok=True)

  list_of_imgs = get_img_list(args.test_csv)
  print( "Testing on %s" % args.test_csv )
  run(model, list_of_imgs, out_folder, args.vis_normal)	

