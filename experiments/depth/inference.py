import argparse
import os
import torch
import numpy as np

import glob

from torch.utils import data
from torch.autograd import Variable
import cv2

from HourglassNetwork import HourglassNetwork
from ReDWebNetReluMin import ReDWebNetReluMin, ReDWebNetReluMin_raw

from ReDWebNet import resNet_data_preprocess
from tqdm import tqdm

from utils import load_obj, save_obj



# code for scale invaraint metric depth evaluation
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vis_depth(depths, colors, i, target_res):
	out = depths.copy()
	out = out - np.min(out)
	out = out / np.max(out) * 255.0
	

	out_color = colors[0].transpose(1, 2, 0)  		
	out_color[:,:,0] = (out_color[:,:,0] * 0.229 + 0.485 ) *255.0 
	out_color[:,:,1] = (out_color[:,:,1] * 0.224 + 0.456 ) *255.0 
	out_color[:,:,2] = (out_color[:,:,2] * 0.225 + 0.406 ) *255.0 
	img = np.zeros((out_color.shape[0],out_color.shape[1]*2,3), np.uint8)
	img[:,:out_color.shape[1], :] = out_color
	img[:,out_color.shape[1] : out_color.shape[1]*2, 0] = out
	img[:,out_color.shape[1] : out_color.shape[1]*2, 1] = out
	img[:,out_color.shape[1] : out_color.shape[1]*2, 2] = out

	img = cv2.resize(img, (2*target_res[1], target_res[0]))
	
	cv2.imwrite("./visualize/%d_depth.jpg" % i, img)



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

def run(model, list_of_imgs, b_resnet_prep, out_folder = "", b_vis_depth=False, max_iter = 100000000000):

	assert not model.training
	
	print('Infering... Saving to %s' % out_folder)

	for step, img_fname in tqdm(enumerate(list_of_imgs)):
		if step > max_iter:
			break

		################################################
		color = cv2.imread(img_fname)
		target_res = color.shape[:2]
		color = cv2.resize(color, (320, 240))
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0	
		if b_resnet_prep:
			color = resNet_data_preprocess(color)

		color = np.expand_dims(color, axis = 0)
		inputs = torch.from_numpy(color)	
			

		################################################
		input_var = inputs.to(_device)
		model_out = model(input_var)
		if isinstance(model_out, tuple):
			output_var, focal_pred_var = model_out[0], model_out[1]
			focal_pred = focal_pred_var.cpu().detach().item()
		else:
			output_var = model_out
			focal_pred = None
		


		################################################
		pred_np = output_var.cpu().detach().numpy()[0,0,:,:]
		
				
	
		if b_vis_depth:
			vis_depth(pred_np, input_var.cpu().detach().numpy(), step, target_res)


		################################################
		img_id = os.path.basename(img_fname).replace(".png", "")
		
		out_dict = {"pred_depth": pred_np, "pred_focal": focal_pred}		
		np.save("%s/%s.npy" % (out_folder, img_id), out_dict)
		# print(step, "Done saving to %s/%s.npy" % (out_folder, img_id))


		################################################
		# this None assignment is necessary to keep the gpu memory clean
		input_var = None
		output_var = None
		inputs = None
		target_res = None

## Usage
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_csv', '-t', default=None)
	parser.add_argument('--model_file', '-model', default=None)	
	parser.add_argument('--max_iter', '-iter', type = int, default=1000000000)	
	parser.add_argument('--out_dir', '-out_dir', type = str, default=None)	
	parser.add_argument('--vis_depth',  '-vis', action='store_true', default=False)

	args = parser.parse_args()
	training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args.model_file)), 'args.pkl'))

	print( "#######################################################################")
	print( 'Testing args:', args	)
	print( "Training args:", training_args	)
	print( "#######################################################################\n\n\n")


	if args.vis_depth:
		os.makedirs('./visualize', exist_ok=True)

	out_folder = args.model_file.replace(".bin", "")
	if args.out_dir:
		out_folder = args.out_dir
	os.makedirs('%s' % out_folder, exist_ok=True)


	NetworkType = {"NIPS":HourglassNetwork, 
				   "ReDWebNetReluMin": ReDWebNetReluMin,
				   "ReDWebNetReluMin_raw": ReDWebNetReluMin_raw,
				  }

	###############################################
	# list_of_imgs = glob.glob('%s/*.png' % args.folder)
	# print( "Testing on %s" % args.folder )
	list_of_imgs = get_img_list(args.test_csv)
	print( "Testing on %s" % args.test_csv )


	###############################################
	b_resnet_prep = training_args['model_name'] != 'NIPS'
	model = NetworkType[training_args['model_name']]().cuda()		
	model.load_state_dict(torch.load(args.model_file))	
	model.eval()

	###############################################	
	run(model, list_of_imgs, b_resnet_prep, out_folder, args.vis_depth, args.max_iter)	
