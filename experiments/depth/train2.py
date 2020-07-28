import argparse
import os

import cv2
import math
import torch
import torch.nn.parallel
import numpy as np

import valid2

import config
import TBLogger

from utils import makedir_if_not_exist, StoreDictKeyPair, save_obj
from torch import optim
from torch.utils import data
from torch.autograd import Variable


from HourglassNetwork import HourglassNetwork
from ReDWebNetReluMin import ReDWebNetReluMin, ReDWebNetReluMin_raw
from LocalBackprojLoss2 import LocalBackprojLoss2
from OASISDataset2 import OASISDataset, OASISDatasetVal, OASIS_collate_fn


def save_model(optimizer, model, iter, prev_iter, prefix=''):
	makedir_if_not_exist(config.JOBS_MODEL_DIR)
	torch.save(model.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%smodel_iter_%d.bin' % (prefix, iter + prev_iter) ))
	torch.save(optimizer.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%sopt_state_iter_%d.bin' % (prefix, iter + prev_iter) ))

def get_prev_iter(pretrained_file):	
	temp = pretrained_file.replace('.bin', '')
	prev_iter = int(temp.split('_')[-1])
	 
	return prev_iter


def vis_depth_by_surface(depths, surface_id):
	print(np.unique(depths))
	out = depths.copy()

	for id in np.unique(surface_id):
		if id == 0:
			continue
		mask = surface_id == id
		out[mask] = out[mask] - np.min(out[mask])
		out[mask] = out[mask] / np.max(out[mask]) * 255.0
	
	return out.astype(np.uint8)

def vis_depth(depths, mask):
	print(np.unique(depths))
	out = depths.copy()

	out[mask] = out[mask] - np.min(out[mask])
	out[mask] = out[mask] / np.max(out[mask]) * 255.0
	
	return out.astype(np.uint8)

def vis_depth_full(depths, mask = None):
	# print(np.unique(depths))
	out = depths.copy()
	if mask is None:
		mask = out > 0
	out = out - np.min(out[mask])
	out = out / np.max(out[mask]) * 255.0
	out[out>255.0] = 255.0
	out[out<0.0] = 0.0
	return out.astype(np.uint8)	


def train(dataset_name, model_name, loss_name, 
		  n_GPUs, b_oppi, b_data_aug, b_sort, \
		  train_file, valid_file,\
		  learning_rate, num_iters, num_epoches,\
		  batch_size, num_loader_workers, pretrained_file,\
		  model_save_interval, model_eval_interval, exp_name):

	NetworkType = {
				   	"NIPS":HourglassNetwork, 
				   	"ReDWebNetReluMin": ReDWebNetReluMin,
				   	"ReDWebNetReluMin_raw": ReDWebNetReluMin_raw,
				  }
	LossType = {
						"LocalBackprojLoss2": LocalBackprojLoss2,
					}
	DatasetsType = {
						"OASISDataset":{'train_dataset':OASISDataset, 'val_dataset':OASISDatasetVal, 't_val_dataset':OASISDataset},
					}

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Using CUDA:", torch.cuda.is_available())
	# create (and load) model. Should wrap with torch.nn.parallel.DistributedDataParallel before loading pretraiend model (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
	model = NetworkType[model_name]().to(device)
	
	b_resnet_prep = model_name != 'NIPS'

	if n_GPUs > 1:
		print( "######################################################")
		print( "Using %d GPUs, batch_size is %d" % (n_GPUs, batch_size))
		print( "######################################################")
		model = torch.nn.parallel.DataParallel(model)

	print ('num_loader_workers:', num_loader_workers)

	# resume from a checkpoint model
	prev_iter = 0
	if pretrained_file:
		model.load_state_dict(torch.load( pretrained_file ))
		prev_iter = get_prev_iter(pretrained_file)
	print ("Prev_iter: {}".format(prev_iter))

	# set up criterion and optimizer
	criterion = LossType[loss_name]()
	
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)	

	try:
		if pretrained_file:
			print (pretrained_file)
			optimizer.load_state_dict(torch.load( pretrained_file.replace('model_', 'opt_state_') ))
	except:
		print("Exception happens when trying to load optimizer state, possibly due to different learning rate strategy.")

	
	# create dataset	
	t_dataset = DatasetsType[dataset_name]['train_dataset']( csv_filename= train_file, b_data_aug = b_data_aug, b_resnet_prep = b_resnet_prep, b_oppi = b_oppi )
	v_dataset = DatasetsType[dataset_name]['val_dataset']( csv_filename= valid_file, b_resnet_prep = b_resnet_prep )	
	tv_dataset = DatasetsType[dataset_name]['t_val_dataset']( csv_filename= train_file, b_resnet_prep = b_resnet_prep )
	t_data_loader = data.DataLoader(t_dataset, batch_size=batch_size, num_workers=num_loader_workers, shuffle=True, collate_fn = OASIS_collate_fn)
	tv_data_loader = data.DataLoader(tv_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = OASIS_collate_fn)
	v_data_loader = data.DataLoader(v_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = OASIS_collate_fn)
	

	# create tensorboard logger
	logger = TBLogger.TBLogger(makedir_if_not_exist(config.JOBS_LOG_DIR))

	cv2.setNumThreads(0)
	
	iter = 1
	best_v_WKDR = float('inf')
	best_siv = float('inf')
	for epoch in range(num_epoches):
		print ("==============epoch = ", epoch)
		for step, (inputs, metric_depth, surface_ids, target, _, focals, names) in enumerate(t_data_loader):

			if iter >= num_iters:
				break
			
			###### zero gradient
			optimizer.zero_grad()

			###### read in training data
			input_var = inputs.to(device)
			metric_depth_var = [a.to(device) for a in metric_depth]
			surface_ids_var = surface_ids.to(device)
			focals_gt_var = focals.to(device)	# TODO
			target_var = [a.to(device) for a in target]

			###### forwarding
			output_var, focal_pred_var = model(input_var)
			
			# TODO: remove
			if iter % 3000 == 0 and dataset_name != 'DIWDataset' :	
				try:			
					# pred_depth = np.exp(output_var.cpu().detach().numpy())	# when the network is predicting log depth.
					pred_depth = output_var.cpu().detach().numpy()	# when the network is predicting absolute depth
					
					c = surface_ids.cpu().detach().numpy()
					_p_img = vis_depth_by_surface(pred_depth[0,0,:,:], c[0,0,:,:])
					_p_full_img = vis_depth(pred_depth[0,0,:,:], c[0,0,:,:] > 0)
					
					logger.add_image('train/pred_depth_surface', torch.from_numpy(_p_img), (iter + prev_iter), dataformats="HW")
					logger.add_image('train/pred_depth', torch.from_numpy(_p_full_img), (iter + prev_iter), dataformats="HW")
					if b_resnet_prep:
						print("ResNet Prep")
						out_color = inputs[0].cpu().detach().numpy()
						out_color[0,:,:] = (out_color[0,:,:] * 0.229 + 0.485 ) *255.0 
						out_color[1,:,:] = (out_color[1,:,:] * 0.224 + 0.456 ) *255.0 
						out_color[2,:,:] = (out_color[2,:,:] * 0.225 + 0.406 ) *255.0 
						out_color = out_color.astype(np.uint8)
						logger.add_image('train/img', torch.from_numpy(out_color), (iter + prev_iter), dataformats="CHW")
					else:
						logger.add_image('train/img', inputs[0], (iter + prev_iter), dataformats="CHW")
					try:
						b = metric_depth[0].cpu().detach().numpy()	
						_gt_img = vis_depth_full(b, c[0,0,:,:] > 0)
						logger.add_image('train/gt_depth', torch.from_numpy(_gt_img), (iter + prev_iter), dataformats="HW")
					except:
						b = np.zeros((240,320), dtype= np.uint8)
						logger.add_image('train/gt_depth', torch.from_numpy(b), (iter + prev_iter), dataformats="HW")
						print("No data for gt depth.")
				except Exception as e:
					print(str(e))

			###### get loss
			if loss_name in ["LocalBackprojLoss", "LocalBackprojLoss2", "BackprojLoss", "BackprojLoss2" ]:
				loss = criterion(preds = output_var, 							 
								gts = metric_depth_var, 
								surface_ids = surface_ids_var,
								focal_gts = focals_gt_var,
								focal_preds = focal_pred_var)			
			
			
			print(iter + prev_iter, "Total_loss: %g" % loss.item())
			if math.isnan(loss.item()):
				import sys
				sys.exit()
			if loss.item() > 1e+8:
				print(names)

			
			###### save to log
			logger.add_value('train/Loss',      loss.item(),      step=(iter + prev_iter) )			
			

			###### back propagate			
			loss.backward()
			optimizer.step()

			
			if (iter + prev_iter) % model_save_interval == 0:
				save_model(optimizer, model, iter, prev_iter)

			if (iter + prev_iter) % model_eval_interval == 0:				
				print ("Evaluating at iter %d" % iter)
				model.eval()
				if n_GPUs > 1:		
					print ("========================================validation set")
					v_rel_error, _, _, v_LSIVRMSE = valid2.valid(model.module, v_data_loader, criterion, in_thresh=0.0)
					print ("========================================training set")
					t_rel_error, _, _, t_LSIVRMSE = valid2.valid(model.module, tv_data_loader, criterion, in_thresh=0.0, max_iter=500)
				else:
					print ("========================================validation set")
					v_rel_error, _, _, v_LSIVRMSE = valid2.valid(model, v_data_loader, criterion, in_thresh=0.0, max_iter=500)
					print ("========================================training set")
					t_rel_error, _, _, t_LSIVRMSE = valid2.valid(model, tv_data_loader, criterion, in_thresh=0.0, max_iter=500)
				
				logger.add_value('train/WKDR', t_rel_error['WKDR_neq'], step=(iter + prev_iter))
				logger.add_value('train/LSIV_RMSE', t_LSIVRMSE["LSIV"], step=(iter + prev_iter))
				logger.add_value('val/WKDR', v_rel_error['WKDR_neq'], step=(iter + prev_iter) )
				logger.add_value('val/LSIV_RMSE', v_LSIVRMSE["LSIV"], step=(iter + prev_iter))
				model.train()
				
				if best_v_WKDR > v_rel_error['WKDR_neq']:
					best_v_WKDR = v_rel_error['WKDR_neq']
					save_model(optimizer, model, iter, prev_iter, prefix = 'best_rel')
				if best_siv > v_LSIVRMSE["LSIV"]:
					best_siv = v_LSIVRMSE["LSIV"]
					save_model(optimizer, model, iter, prev_iter, prefix = 'best_siv')
				
				save_model(optimizer, model, iter, prev_iter)
					
			iter += 1

			inputs = None
			target = None			

		if iter >= num_iters:
			break

	

	save_model(optimizer, model, iter, prev_iter)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--network_name', '-nn', default=config.DEFAULT_NETWORK_NAME)
	parser.add_argument('--train_file', '-t', default='')	# should be absolute path
	parser.add_argument('--valid_file', '-v', default='')
	parser.add_argument('--dataset_name', '-dn', default='OASISDataset') 
	parser.add_argument('--model_name', '-mn', default='NIPS') #
	parser.add_argument('--loss_name', default='BackprojLoss') # 
	# parser.add_argument('--optim_name', '-on', default=config.DEFAULT_OPTIM_NAME)
	parser.add_argument('--num_iters', '-iter', default=100000, type=int)
	parser.add_argument('--num_epoches', '-ne', default=100000, type=int)
	parser.add_argument('--batch_size', '-bs', default=4, type=int)
	parser.add_argument('--model_save_interval', '-mt', default=2000, type=int)
	parser.add_argument('--model_eval_interval', '-et', default=3000, type=int)
	parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
	parser.add_argument('--n_GPUs', '-ngpu', default=1, type=int)	
	parser.add_argument('--num_loader_workers', '-nlw', type=int, default=2)
	parser.add_argument('--pretrained_file', '-pf', default=None)
	parser.add_argument('--b_oppi', '-b_oppi', action='store_true', default=False)
	parser.add_argument('--b_sort', '-b_sort', action='store_true', default=False)
	parser.add_argument('--b_data_aug', '-b_data_aug', action='store_true', default=False)
	# parser.add_argument('--debug', '-d', action='store_true')
	parser.add_argument('--exp_name', default='debug') #

	args = parser.parse_args()

	args_dict = vars(args)

	config.JOBS_MODEL_DIR = "./exp/%s/models" % args.exp_name
	config.JOBS_LOG_DIR = "./exp/%s/log" % args.exp_name
	config.JOBS_DIR = './exp/%s' % args.exp_name

	folder = makedir_if_not_exist(config.JOBS_DIR)
	save_obj(args_dict, os.path.join(config.JOBS_DIR, 'args.pkl'))
	
	train(**args_dict)

	print ("End of train.py")






