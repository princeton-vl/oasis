import argparse
import os
import cv2
import torch
import numpy as np


import valid2
from utils import save_obj, load_obj, makedir_if_not_exist

from HourglassNetwork import HourglassNetwork
from ReDWebNet import ReDWebNet_resnet50
from OASISDataset2 import OASISDataset, OASISDatasetVal, OASISDatasetDIWVal, OASIS_collate_fn
from ReDWebNetReluMin import ReDWebNetReluMin, ReDWebNetReluMin_raw
from torch.utils import data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_file', '-t', default='')
	parser.add_argument('--verbal', '-verbal', action='store_true', default=False)
	parser.add_argument('--num_iters', '-iter', default=100000, type=int)
	parser.add_argument('--output_file', '-o', default=None)
	parser.add_argument('--vis_depth', '-vis', action='store_true', default=False)
	###################################################################################
	# for evaluating trained models
	parser.add_argument('--model_file', '-model', default=None)	
	parser.add_argument('--DIW_rel_depth', action='store_true', default=False, help="If set true, load the DIW style relative depth.")	
	###################################################################################
	# for evaluating pred already generated
	parser.add_argument('--pred_path', '-pred_path', default=None)
	parser.add_argument('--fixed_focal_length', '-fixed_focal_length', nargs='+', type=float)
	parser.add_argument('--plane', '-plane', action='store_true', default = False)
	parser.add_argument('--b_keep_neg', '-b_keep_neg', action='store_true', default = False, 
						help="Keep the negative depth in the predicted depth. Used for relative depth trained networks.")

	args = parser.parse_args()
	print( "#######################################################################")
	print( 'Testing args:', args	)
	if args.vis_depth:
		if not os.path.exists('./visualize'):
			os.makedirs('./visualize')

	if args.pred_path is None:
		training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args.model_file)), 'args.pkl'))	
		print( "Training args:", training_args	)
		print( "#######################################################################\n\n\n")

		b_resnet_prep = {
						"NIPS":False, 
						"ReDWebNetReluMin": True,
						"ReDWebNetReluFixed": True,
						"ReDWebNetReluMin_raw": True,
		}[training_args['model_name']]
	else:
		b_resnet_prep = True
	
	print("=====> b_resnet_prep: %s" % b_resnet_prep)


	if args.DIW_rel_depth:
		test_dataset = OASISDatasetDIWVal(csv_filename = args.test_file, b_resnet_prep = b_resnet_prep )
		test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = OASIS_collate_fn)
	else:
		test_dataset = OASISDatasetVal(csv_filename = args.test_file, b_resnet_prep = b_resnet_prep )
		test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = OASIS_collate_fn)
	print( "Testing on %s" % args.test_file	)


	if args.pred_path is None:
		NetworkType = {
						"NIPS":HourglassNetwork, 
						"ReDWebNetReluMin": ReDWebNetReluMin,
						"ReDWebNetReluMin_raw": ReDWebNetReluMin_raw,
					  }
		model = NetworkType[training_args['model_name']]().cuda()
		if training_args['n_GPUs'] > 1:
			model = torch.nn.parallel.DataParallel(model)
			model.load_state_dict(torch.load(args.model_file))	
			model = model.module
		else:		
			model.load_state_dict(torch.load(args.model_file))	

		model.eval()	

		test_rel_error, test_metric_error, metric_error_gt, metric_error_gt_no_order = \
														valid2.valid(model = model, 
																	data_loader = test_data_loader, 
																	criterion = None, 
																	max_iter = args.num_iters, 
																	verbal=True, 
																	b_vis_depth=args.vis_depth, 
																	in_thresh=0.0,
																	b_eval_rel_depth_only = args.DIW_rel_depth)	
		model.train()
		run_log = "success"
	else:		
		if args.fixed_focal_length is None:
			args.fixed_focal_length = [None]
		for focal in args.fixed_focal_length:
			print("Operating with tentative focal length:",  focal)

			test_rel_error, metric_error_gt_no_order, run_log = \
											valid2.valid_known(pred_folder = args.pred_path, 
																data_loader = test_data_loader, 
																max_iter = args.num_iters, 
																b_plane = args.plane,
																fixed_focal_len = focal,
																verbal = args.verbal, 
																b_vis_depth=args.vis_depth,
																b_eval_rel_depth_only = args.DIW_rel_depth,
																b_keep_neg=args.b_keep_neg)	


	if args.output_file is not None:		
		save_obj({'test_rel_error':test_rel_error, 'metric_error_gt_no_order':metric_error_gt_no_order, "run_log": run_log}, args.output_file)


