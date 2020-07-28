import torch
import math
from torch.autograd import Variable
import numpy as np

class LocalBackprojLoss2(object):
	def __init__(self, height = 240, width = 320):

		print ("Using Local Back Projection Loss v2")
		print ("	==> Align prediction to gt.")
		
		xs = np.linspace(0, width-1, width)
		ys = np.linspace(0, height-1, height)
		self.xv, self.yv = np.meshgrid(xs, ys)
		
		self.xv -= 0.5 * width
		self.yv -= 0.5 * height

		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.xv = torch.tensor(self.xv, dtype = torch.float32, device = device)
		self.yv = torch.tensor(self.yv, dtype = torch.float32, device = device)


	def back_proj(self, depth, focal, surface_id, b_normalize = False):
		
		mask = surface_id > 0 
		
		X = torch.mul(self.xv[mask] / (focal + 1e-8), depth[mask])
		Y = torch.mul(self.yv[mask] / (focal + 1e-8), depth[mask])
		Z = depth[mask]

		if b_normalize:
			std_X = torch.std(X) + 1e-8

			X = X / std_X
			Y = Y / std_X
			Z = Z / std_X

		surface_id = surface_id[mask]
		return X, Y, Z, surface_id

	def loss_func(self, pred, gt, surface_id, focal_pred, focal_gt):
		'''
			pred 	   : HxW
			gt   	   : HxW
			surface_id : HxW
			Aligns gt to prediction.
		'''

		total_loss = torch.tensor(0.0, dtype = torch.float32, device = pred.device)

		_x_gt,   _y_gt,   _z_gt,   _           = self.back_proj(gt,   focal_gt,   surface_id, b_normalize = False)
		_x_pred, _y_pred, _z_pred, _surface_id = self.back_proj(pred, focal_pred, surface_id, b_normalize = True)


		for s_id in torch.unique(_surface_id):			
			if s_id.item() == 0:
				continue

			mask = (_surface_id == s_id) & (_z_pred > 0)
			
			x_gt,   y_gt,   z_gt   = _x_gt[mask],   _y_gt[mask],   _z_gt[mask]
			x_pred, y_pred, z_pred = _x_pred[mask], _y_pred[mask], _z_pred[mask]

			N_pt = x_gt.shape[0] + 1e-8
			if N_pt < 10:
				# import pdb; pdb.set_trace()
				continue

			x_gt_2 = torch.sum(torch.pow(x_gt, 2))
			y_gt_2 = torch.sum(torch.pow(y_gt, 2))
			z_gt_2 = torch.sum(torch.pow(z_gt, 2))
			
			z_gt_sum = torch.sum(z_gt)
			z_pred_sum = torch.sum(z_pred)

			x_gtx_pred = torch.sum(torch.mul(x_gt, x_pred))
			y_gty_pred = torch.sum(torch.mul(y_gt, y_pred))
			z_gtz_pred = torch.sum(torch.mul(z_gt, z_pred))

			denominator = x_gt_2 + y_gt_2 + z_gt_2 - z_gt_sum * z_gt_sum / N_pt + 1e-8
			scale2 =(x_gtx_pred + y_gty_pred + z_gtz_pred - z_gt_sum * z_pred_sum / N_pt) / denominator
			delta2 = (z_pred_sum - scale2 * z_gt_sum) / N_pt


			# if torch.isnan(scale2) or torch.isnan(delta2) or torch.isnan(denominator):
			# 	import pdb; pdb.set_trace()

			total_loss += torch.sum(torch.pow(scale2 * x_gt - x_pred, 2))
			total_loss += torch.sum(torch.pow(scale2 * y_gt - y_pred, 2))
			total_loss += torch.sum(torch.pow(scale2 * z_gt + delta2 - z_pred, 2))

		return total_loss
	
	def __call__(self, preds, gts, surface_ids, focal_preds, focal_gts):
		'''
		preds:       nSamples x 1 x Height x Width.  Predicted depth.
		gts:         a list of nbatch elements, tensors of Height x Width.  Ground truth depth 
		surface_ids: nSamples x 1 x Height x Width, value starts from 0.
		                 0 denotes invalid, 1, 2, 3 ... are the surface ids of pixels
		focal_preds: nSamples x 1 tensor
		focal_gts:   nSamples x 1 tensor

		This loss calculates the loss of surface-wise aligning the prediction to the gt.
		'''

		assert(isinstance(gts, list))		
		
		total_loss = torch.tensor(0.0, dtype = torch.float32, device = preds.device)
		
		example_count = 1e-8
		for _idx, gt in enumerate(gts):
		
			if gt.nelement() == 0:
				continue
			
			pred = preds[_idx, 0, ...]			
			surface_id = surface_ids[_idx, 0, ...]
			
			focal_pred = focal_preds[_idx]
			focal_gt = focal_gts[_idx]

			# pay attention to how the gt and pred are sent into the loss function
			_loss = self.loss_func( pred = gt, 
                                    gt = pred, 
                                    surface_id = surface_id, 
                                    focal_pred = focal_gt, 
                                    focal_gt = focal_pred)

			total_loss = total_loss + _loss	
			
			example_count += 1.0
		
		return total_loss / example_count

