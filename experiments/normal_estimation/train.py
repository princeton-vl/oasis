import argparse
import os
import cv2

import torch
import torch.nn.parallel
import numpy as np

import math
import valid
import sys

import common.config as config
import common.TBLogger as TBLogger

from common.utils import makedir_if_not_exist, StoreDictKeyPair, save_obj, load_obj
from torch import optim
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

from common.models.NIPSNetwork import NIPSSurfaceNetwork
from common.losses.CosineAngularLoss import CosineAngularLoss

from common.datasets.OASISDataset import OASISNormalDataset, OASISNormalDatasetVal

def save_model(optimizer, model, iter, prev_iter, prefix=''):
  makedir_if_not_exist(config.JOBS_MODEL_DIR)
  torch.save(model.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%smodel_iter_%d.bin' % (prefix, iter + prev_iter) ))
  torch.save(optimizer.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%sopt_state_iter_%d.bin' % (prefix, iter + prev_iter) ))

def get_prev_iter(pretrained_file): 
  temp = pretrained_file.replace('.bin', '')
  prev_iter = int(temp.split('_')[-1])
   
  return prev_iter

def train(dataset_name, model_name, loss_name, in_coord_sys, out_coord_sys, n_GPUs, data_aug,
      train_file, valid_file, learning_rate, num_iters, num_epoches, batch_size, num_loader_workers, 
      pretrained_model, pretrained_optimizer, model_save_interval, model_eval_interval, scratch, exp_name):

  print(model_name)
  NetworkType = {'NIPSSurface':NIPSSurfaceNetwork}
  LossType = {'CosineAngularLoss':CosineAngularLoss}
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # create (and load) model.
  model = NetworkType[model_name]().to(device)
  n_GPUs = torch.cuda.device_count()
  if n_GPUs > 1:
    print("######################################################")
    print("Using %d GPUs, batch_size is %d" % (n_GPUs, batch_size))
    print("######################################################")
    model = torch.nn.DataParallel(model)

  print('num_loader_workers: {}'.format(num_loader_workers))

  # resume from a checkpoint model
  prev_iter = 0
  if pretrained_model:
    print('Finetuning: {}'.format(pretrained_model))
    state_dict = torch.load(pretrained_model)
    model.load_state_dict(state_dict)
    prev_iter = get_prev_iter(pretrained_model)
  print("Prev_iter: {}".format(prev_iter))

  # set up optimizer
  optimizer = optim.RMSprop(model.parameters(), lr=learning_rate) 
    
  if pretrained_optimizer:
    print('Loading optimizer: {}'.format(pretrained_optimizer))
    state_dict = torch.load(pretrained_optimizer)
    optimizer.load_state_dict(state_dict)
  
  # register dataset type 
  DatasetsType = {
                  "OASISNormalDataset":{'train_dataset':OASISNormalDataset, 'val_dataset':OASISNormalDatasetVal, 't_val_dataset':OASISNormalDatasetVal}
                  }

  collate_fn = default_collate
  t_dataset = DatasetsType[dataset_name]['train_dataset'](csv_filename=train_file, data_aug = data_aug)
  v_dataset = DatasetsType[dataset_name]['val_dataset'](csv_filename=valid_file)
  tv_dataset = DatasetsType[dataset_name]['t_val_dataset'](csv_filename=train_file)

  # create dataset  
  t_data_loader = data.DataLoader(t_dataset, batch_size=batch_size, num_workers=num_loader_workers, shuffle=True, collate_fn = collate_fn)
  tv_data_loader = data.DataLoader(tv_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = collate_fn)
  v_data_loader = data.DataLoader(v_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = collate_fn)

  # set up criterion
  criterion = LossType[loss_name]()
  
  # create tensorboard logger
  logger = TBLogger.TBLogger(makedir_if_not_exist(config.JOBS_LOG_DIR))


  cv2.setNumThreads(0)

  # Create coordinate system change vector
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
  else:
    raise Exception("Error in train file")

  iter = 1
  best_v_MAE = 1000
  for epoch in range(num_epoches):
    print("==============epoch = {}".format(epoch))
    for step, data_tuple in enumerate(t_data_loader):
      if iter >= num_iters:
        break

      ###### zero gradient
      optimizer.zero_grad()

      ###### read in training data
      if 'NYU' in train_file or 'SNOW' in train_file:
        (inputs, target, input_res) = data_tuple
        target_var = target.to(device)
      elif 'OASIS' in train_file:
        (inputs, target, mask, input_res) = data_tuple
        target_var = target.to(device)
        mask_var = mask.to(device)
      else:
        raise Exception("Error in train file")

      input_var = inputs.to(device)

      ###### forwarding
      output_var = model(input_var)
      # Change coordinate systems
      output_var[:,0,:,:] *= coord_change[0]
      output_var[:,1,:,:] *= coord_change[1]
      output_var[:,2,:,:] *= coord_change[2]

      if 'NYU' in train_file:
        # Resize output to batchsize x 3 x 480 x 640
        output = torch.nn.functional.interpolate(output_var, size=(480, 640), mode='bilinear')
        # Only give the cropped portion to compare (b/c Matlab cropped the normals but network makes full prediction)
        # 427 x 561
        loss = criterion(output[:,:,44:471, 40:601], target_var)

      elif 'SNOW' in train_file:
        truth_normals = target[:,:3].to(device)
        coords = target[:,3:].long() # needs to be Long tensor for indexing
        # Get normals using coordinates (one per image)
        pred_normals = torch.Tensor().to(device)
        # Might not have exactly batch size elements
        output = output_var#.clone().cpu().detach().numpy()
        for i in range(0, coords.shape[0]):
          # Coordinates in original resolution
          # Resize output to batchsize x 3 x orig_width x orig_height
          x = coords[i,0]
          y = coords[i,1]
          orig_height = input_res[0][i]
          orig_width = input_res[1][i]
          output = torch.nn.functional.interpolate(output, size=(orig_height, orig_width), mode='bilinear')
          pred_normals = torch.cat((pred_normals, output[i,:,x,y]), -1)
        pred_normals = torch.reshape(pred_normals, (coords.shape[0], 3)).double().to(device)
        ###### get loss
        loss = criterion(pred_normals, truth_normals)

      elif 'OASIS' in train_file:
        loss = 0.0
        total_pixels = 0
        # Don't interpolate during training. Only during validation evaluate at full resolution
        for i in range(0, len(input_res[0])):
          orig_height = input_res[0][i]
          orig_width = input_res[1][i]
          output = output_var
          target = target_var
          mask = mask_var
          # torch interpolate requires 4D vector. So index afterward
          # output = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='bilinear')
          # target = torch.nn.functional.interpolate(target_var, size=(orig_height, orig_width), mode='bilinear')
          # mask = torch.nn.functional.interpolate(mask_var, size=(orig_height, orig_width), mode='bilinear')
          mask = mask.byte().squeeze(1) # remove color channel
          mask = mask[i,:,:]
          if torch.sum(mask) > 0:
            output = output[i,:,:,:].permute(1,2,0)[mask,:].float()
            target = target[i,:,:,:].permute(1,2,0)[mask,:].float()
            total_pixels += torch.sum(mask)
            loss += criterion(output, target) * torch.sum(mask)
        loss /= total_pixels
      
      else:
        raise Exception("Error in train file")
        
      print("Epoch {} Iter {}: {}".format(epoch, iter + prev_iter, loss.data))
      sys.stdout.flush()

      ###### back propagate     
      loss.backward()
      optimizer.step()

      ###### save to log
      # need to unload Tensor to CPU so that it doesn't complain
      logger.add_value('Training Loss', loss.data.cpu(), step=(iter + prev_iter) )


      if (iter + prev_iter) % model_save_interval == 0:
        save_model(optimizer, model, iter, prev_iter)

      if (iter + prev_iter) % model_eval_interval == 0:       
        print("Evaluating at iter %d" % iter)
        model.eval()
        if n_GPUs > 1:    
          print("========================================validation set")
          v_rel_error = valid.valid(model.module, in_coord_sys, out_coord_sys, v_data_loader, 'OASISNormalDatasetVal', in_thresh=0.0)
          print("========================================training set")
          t_rel_error = valid.valid(model.module, in_coord_sys, out_coord_sys, tv_data_loader, 'OASISNormalDatasetVal', in_thresh=0.0, max_iter=500)
        else:
          print("========================================validation set")
          v_rel_error = valid.valid(model, in_coord_sys, out_coord_sys, v_data_loader,'OASISNormalDatasetVal', in_thresh=0.0, max_iter=500)
          print("========================================training set")
          t_rel_error = valid.valid(model, in_coord_sys, out_coord_sys, tv_data_loader, 'OASISNormalDatasetVal', in_thresh=0.0, max_iter=500)
        if True:
          logger.add_value('Val Mean Angle Error', v_rel_error['MAE'], step=(iter + prev_iter))
          logger.add_value('Val < 11.25', v_rel_error['11.25'], step=(iter + prev_iter))
          logger.add_value('Train Mean Angle Error', t_rel_error['MAE'], step=(iter + prev_iter))
          logger.add_value('Train < 11.25', t_rel_error['11.25'], step=(iter + prev_iter))
          logger.add_value('Val < 22.5', t_rel_error['22.5'], step=(iter + prev_iter))
          logger.add_value('Train < 22.5', t_rel_error['22.5'], step=(iter + prev_iter))
          logger.add_value('Val < 30', t_rel_error['30'], step=(iter + prev_iter))
          logger.add_value('Train < 30', t_rel_error['30'], step=(iter + prev_iter))
          logger.add_value('Val Median Angle Error', v_rel_error['MDAE'], step=(iter + prev_iter))
          logger.add_value('Train Median Angle Error', t_rel_error['MDAE'], step=(iter + prev_iter))

        # Save best validation model
        if v_rel_error['MAE'] < best_v_MAE:
          best_v_MAE = v_rel_error['MAE']
          save_model(optimizer, model, iter, prev_iter, prefix='best_')
          print("Best validation error thus far: {}".format(best_v_MAE))

        model.train()
        save_model(optimizer, model, iter, prev_iter)
          
      iter += 1

      inputs = None
      target = None
      input_res = None

    if iter >= num_iters:
      break  

  save_model(optimizer, model, iter, prev_iter)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', '-t', default='')
  parser.add_argument('--valid_file', '-v', default='')
  parser.add_argument('--dataset_name', '-dn', default='')
  parser.add_argument('--model_name', '-mn', default='NIPS')
  parser.add_argument('--loss_name', default='') 
  parser.add_argument('--in_coord_sys', '-in_coord_sys', type=str, required=True)
  parser.add_argument('--out_coord_sys', '-out_coord_sys', type=str, required=True)
  parser.add_argument('--num_iters', '-iter', default=100000, type=int)
  parser.add_argument('--num_epoches', '-ne', default=100000, type=int)
  parser.add_argument('--batch_size', '-bs', default=4, type=int)
  parser.add_argument('--model_save_interval', '-mt', default=5000, type=int)
  parser.add_argument('--model_eval_interval', '-et', default=3000, type=int)
  parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
  parser.add_argument('--n_GPUs', '-ngpu', default=1, type=int) 
  parser.add_argument('--num_loader_workers', '-nlw', type=int, default=2)
  parser.add_argument('--pretrained_model', '-pm', default=None)
  parser.add_argument('--pretrained_optimizer', '-po', default=None)
  parser.add_argument('--scratch', '-scratch', action='store_true', default=False)
  parser.add_argument('--data_aug', '-data_aug', action='store_true', default=False)
  parser.add_argument('--exp_name', '-exp_name', default='debug')

  args = parser.parse_args()
  setattr(args, 'n_GPUs', torch.cuda.device_count())
  args_dict = vars(args)

  config.JOBS_MODEL_DIR = "./exp/%s/models" % args.exp_name
  config.JOBS_LOG_DIR = "./exp/%s/log" % args.exp_name
  config.JOBS_DIR = './exp/%s' % args.exp_name

  folder = makedir_if_not_exist(config.JOBS_DIR)
  save_obj(args_dict, os.path.join(config.JOBS_DIR, 'args.pkl'))

  train(**args_dict)

  print("End of train.py")
