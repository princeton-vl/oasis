import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
from distutils.version import LooseVersion
from sacred import Experiment
from sacred.observers import FileStorageObserver
from easydict import EasyDict as edict
import torch
from torch.utils import data
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as tf
from tqdm import tqdm

from models.baseline_same import Baseline as UNet
from utils.loss import hinge_embedding_loss, class_balanced_cross_entropy_loss, surface_normal_loss, parameter_loss
from utils.misc import AverageMeter, get_optimizer
from utils.metric import eval_iou, eval_plane_prediction
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import k_inv_dot_xy1
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss
from match_segmentation import MatchSegmentation
# from plane_dataset import PlaneDataset
from surface_dataset import SurfaceDataset


from torch.utils.tensorboard import SummaryWriter
class TBLogger(object):
	def __init__(self, folder, flush_secs=60):
		self.writer = SummaryWriter(log_dir = folder, flush_secs=flush_secs)
		
	def add_value(self, name, value, step):
		self.writer.add_scalar(tag = name, scalar_value = value, global_step=step)
	def add_image(self, name, value, step, dataformats):
		self.writer.add_image(tag = name, img_tensor = value, global_step=step, dataformats=dataformats)


ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments'))


def load_dataset(subset, cfg):
	transforms = tf.Compose([
		tf.ToTensor(),
		tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	is_shuffle = subset.find("train") >= 0
	#loaders = data.DataLoader(
	#    PlaneDataset(subset=subset, transform=transforms, root_dir=cfg.root_dir),
	#    batch_size=cfg.batch_size, shuffle=is_shuffle, num_workers=cfg.num_workers,
	#)
	loaders = data.DataLoader(
		SurfaceDataset(subset=subset, transform=transforms, root_dir=cfg.root_dir),
		batch_size=cfg.batch_size, shuffle=is_shuffle, num_workers=cfg.num_workers,
	)

	return loaders


@ex.command
def train(_run, _log):
	cfg = edict(_run.config)

	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_dir = os.path.join('experiments', cfg.exp_name, str(_run._id), 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)


	tb_logger = TBLogger(checkpoint_dir)

	# build network
	network = UNet(cfg.model)

	if not cfg.resume_dir == 'None':
		model_dict = torch.load(cfg.resume_dir)
		network.load_state_dict(model_dict)

	# load nets into gpu
	if cfg.num_gpus > 1 and torch.cuda.is_available():
		network = torch.nn.DataParallel(network)
	network.to(device)

	# set up optimizers
	optimizer = get_optimizer(network.parameters(), cfg.solver)

	# data loader
	data_loader = load_dataset(cfg.train_split, cfg.dataset)

	# save losses per epoch
	history = {'losses': [], 'losses_pull': [], 'losses_push': [], 'losses_binary': [], 'losses_depth': [],
			   'ioues': [], 'rmses': []}

	network.train(not cfg.model.fix_bn)

	bin_mean_shift = Bin_Mean_Shift()
	instance_parameter_loss = InstanceParameterLoss()

	lr_schd = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
	
	over_all_iter = 0
	# main loop
	for epoch in range(cfg.num_epochs):
		batch_time = AverageMeter()
		losses = AverageMeter()
		losses_pull = AverageMeter()
		losses_push = AverageMeter()
		losses_binary = AverageMeter()
		losses_depth = AverageMeter()
		losses_normal = AverageMeter()
		losses_instance = AverageMeter()
		ioues = AverageMeter()
		rmses = AverageMeter()
		instance_rmses = AverageMeter()
		mean_angles = AverageMeter()
		
		
		tic = time.time()
		for iter, sample in enumerate(data_loader):
			image = sample['image'].to(device)
			instance = sample['instance'].to(device)
			semantic = sample['semantic'].to(device)
			gt_depth = sample['depth'].to(device)
			gt_seg = sample['gt_seg'].to(device)
			roi = None
			if 'roi' in sample:
				roi = sample['roi'].to(device)
			gt_plane_parameters = sample['plane_parameters'].to(device)
			valid_region = sample['valid_region'].to(device)
			gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)

			# forward pass
			logit, embedding, _, _, param = network(image)

			segmentations, sample_segmentations, sample_params, centers, sample_probs, sample_gt_segs = \
				bin_mean_shift(logit, embedding, param, gt_seg)

			# calculate loss
			loss, loss_pull, loss_push, loss_binary, loss_depth, loss_normal, loss_parameters, loss_pw, loss_instance, loss_boundary = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
			batch_size = image.size(0)
			for i in range(batch_size):
				_loss, _loss_pull, _loss_push = hinge_embedding_loss(embedding[i:i+1], sample['num_planes'][i:i+1],
																	 instance[i:i+1], device)

				_loss_binary = class_balanced_cross_entropy_loss(logit[i], semantic[i], roi[i])

				_loss_normal, mean_angle = surface_normal_loss(param[i:i+1],
															   gt_plane_parameters[i:i+1], valid_region[i:i+1])

				_loss_L1 = parameter_loss(param[i:i + 1],
										  gt_plane_parameters[i:i + 1], valid_region[i:i + 1])
				_loss_depth, rmse, infered_depth = Q_loss(param[i:i+1], k_inv_dot_xy1, gt_depth[i:i+1])

				if segmentations[i] is None:
					continue

				_instance_loss, instance_depth, instance_abs_disntace, _ = \
					instance_parameter_loss(segmentations[i], sample_segmentations[i], sample_params[i],
										valid_region[i:i+1], gt_depth[i:i+1])

				_loss += _loss_binary + _loss_depth + _loss_normal + _instance_loss + _loss_L1
				#_loss += _loss_binary + _loss_depth + _loss_normal + _pw_loss + _instance_loss + _loss_L1

				# planar segmentation iou
				prob = torch.sigmoid(logit[i])
				mask = (prob > 0.5).float().cpu().numpy()
				iou = eval_iou(mask, semantic[i].cpu().numpy())
				ioues.update(iou * 100)
				instance_rmses.update(instance_abs_disntace.item())
				rmses.update(rmse.item())
				mean_angles.update(mean_angle.item())

				loss += _loss
				loss_pull += _loss_pull
				loss_push += _loss_push
				loss_binary += _loss_binary
				loss_depth += _loss_depth
				loss_normal += _loss_normal
				#loss_pw += _pw_loss
				loss_instance += _instance_loss

			loss /= batch_size
			loss_pull /= batch_size
			loss_push /= batch_size
			loss_binary /= batch_size
			loss_depth /= batch_size
			loss_normal /= batch_size
			#loss_pw /= batch_size
			loss_instance /= batch_size

			# Backward
			if torch.isnan(loss) != 0:
				import pdb; pdb.set_trace()
				pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# update loss
			losses.update(loss.item())
			losses_pull.update(loss_pull.item())
			losses_push.update(loss_push.item())
			losses_binary.update(loss_binary.item())
			losses_depth.update(loss_depth.item())
			losses_normal.update(loss_normal.item())
			#losses_pw.update(loss_pw.item())
			losses_instance.update(loss_instance.item())

			# update time
			batch_time.update(time.time() - tic)
			tic = time.time()
			
			
			over_all_iter += 1
			if iter % cfg.print_interval == 0:
				tb_logger.add_value(name = 'Loss', value = losses.val, step = over_all_iter)
				tb_logger.add_value(name = 'Pull', value = losses_pull.val, step = over_all_iter)
				tb_logger.add_value(name = 'Push', value = losses_push.val, step = over_all_iter)
				tb_logger.add_value(name = 'INS', value = losses_instance.val, step = over_all_iter)
				tb_logger.add_value(name = 'Binary', value = losses_binary.val, step = over_all_iter)
				tb_logger.add_value(name = 'IoU', value = ioues.val, step = over_all_iter)
				tb_logger.add_value(name = 'LN', value = losses_normal.val, step = over_all_iter)
				tb_logger.add_value(name = 'AN', value = mean_angles.val, step = over_all_iter)
				tb_logger.add_value(name = 'Depth', value = losses_depth.val, step = over_all_iter)
				tb_logger.add_value(name = 'INSDEPTH', value = instance_rmses.val, step = over_all_iter)
				tb_logger.add_value(name = 'RMSE', value = rmses.val, step = over_all_iter)
				tb_logger.add_value(name = 'LR', value = lr_schd.get_lr()[0], step = over_all_iter)
				

				_log.info(f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
						  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
						  f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
						  f"Pull: {losses_pull.val:.4f} ({losses_pull.avg:.4f}) "
						  f"Push: {losses_push.val:.4f} ({losses_push.avg:.4f}) "
						  f"INS: {losses_instance.val:.4f} ({losses_instance.avg:.4f}) "
						  f"Binary: {losses_binary.val:.4f} ({losses_binary.avg:.4f}) "
						  f"IoU: {ioues.val:.2f} ({ioues.avg:.2f}) "
						  f"LN: {losses_normal.val:.4f} ({losses_normal.avg:.4f}) "
						  f"AN: {mean_angles.val:.4f} ({mean_angles.avg:.4f}) "
						  f"Depth: {losses_depth.val:.4f} ({losses_depth.avg:.4f}) "
						  f"INSDEPTH: {instance_rmses.val:.4f} ({instance_rmses.avg:.4f}) "
						  f"RMSE: {rmses.val:.4f} ({rmses.avg:.4f}) "
						  f"LR: {lr_schd.get_lr()[0]:.6f}\t")

				
			if (iter + 1) % cfg.save_interval == 0:
				torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_iter_{iter}.pt"))

		lr_schd.step() 
		
		_log.info(f"* epoch: {epoch:2d}\t"
			      f"LR: {lr_schd.get_lr()[0]:.6f}\t"
				  f"Loss: {losses.avg:.6f}\t"
				  f"Pull: {losses_pull.avg:.6f}\t"
				  f"Push: {losses_push.avg:.6f}\t"
				  f"Binary: {losses_binary.avg:.6f}\t"
				  f"Depth: {losses_depth.avg:.6f}\t"
				  f"IoU: {ioues.avg:.2f}\t"
				  f"RMSE: {rmses.avg:.4f}\t")

		# save history
		history['losses'].append(losses.avg)
		history['losses_pull'].append(losses_pull.avg)
		history['losses_push'].append(losses_push.avg)
		history['losses_binary'].append(losses_binary.avg)
		history['losses_depth'].append(losses_depth.avg)
		history['ioues'].append(ioues.avg)
		history['rmses'].append(rmses.avg)

		# save checkpoint
		torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
		pickle.dump(history, open(os.path.join(checkpoint_dir, 'history.pkl'), 'wb'))


@ex.command
def eval(_run, _log):
	cfg = edict(_run.config)

	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_dir = os.path.join('experiments', str(_run._id), 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# build network
	network = UNet(cfg.model)

	if not cfg.resume_dir == 'None':
		model_dict = torch.load(cfg.resume_dir)
		network.load_state_dict(model_dict)

	# load nets into gpu
	if cfg.num_gpus > 1 and torch.cuda.is_available():
		network = torch.nn.DataParallel(network)
	network.to(device)
	network.eval()

	# data loader
	#data_loader = load_dataset('val', cfg.dataset)
	data_loader = load_dataset('test', cfg.dataset)

	pixel_recall_curve = np.zeros((13))
	plane_recall_curve = np.zeros((13, 3))

	bin_mean_shift = Bin_Mean_Shift()
	instance_parameter_loss = InstanceParameterLoss()
	match_segmentatin = MatchSegmentation()

	with torch.no_grad():
		for iter, sample in enumerate(tqdm(data_loader)):
			image = sample['image'].to(device)
			instance = sample['instance'].to(device)
			gt_seg = sample['gt_seg'].numpy()
			semantic = sample['semantic'].to(device)
			gt_depth = sample['depth'].to(device)
			gt_plane_parameters = sample['plane_parameters'].to(device)
			valid_region = sample['valid_region'].to(device)
			gt_plane_num = sample['num_planes'].int()
			gt_plane_instance_parameter = sample['plane_instance_parameter'].numpy()

			# forward pass
			logit, embedding, _, _, param = network(image)

			prob = torch.sigmoid(logit[0])

			# infer per pixel depth using per pixel plane parameter
			_, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, gt_depth)

			# fast mean shift
			segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(prob, embedding[0], param, mask_threshold=0.1)

			# since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
			# we thus use avg_pool_2d to smooth the segmentation results
			b = segmentation.t().view(1, -1, 192, 256)
			pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
			b = pooling_b.view(-1, 192*256).t()
			segmentation = b

			# infer instance depth
			instance_loss, instance_depth, instance_abs_disntace, instance_parameter = \
				instance_parameter_loss(segmentation, sampled_segmentation, sample_param,
										valid_region, gt_depth, False)

			# greed match of predict segmentation and ground truth segmentation using cross entropy to better visualzation
			matching = match_segmentatin(segmentation, prob.view(-1, 1), instance[0], gt_plane_num)

			# return cluster results
			predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

			# reindexing to matching gt segmentation for better visualization
			matching = matching.cpu().numpy().reshape(-1)
			used = set([])
			max_index = max(matching) + 1
			for i, a in zip(range(len(matching)), matching):
				if a in used:
					matching[i] = max_index
					max_index += 1
				else:
					used.add(a)
			predict_segmentation = matching[predict_segmentation]

			# mask out non planar region
			predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
			predict_segmentation = predict_segmentation.reshape(192, 256)

			# visualization and evaluation
			h, w = 192, 256
			image = tensor_to_image(image.cpu()[0])
			semantic = semantic.cpu().numpy().reshape(h, w)
			mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
			gt_seg = gt_seg.reshape(h, w)
			depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
			per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

			# use per pixel depth for non planar region
			depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)
			gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)

			# evaluation plane segmentation
			pixelStatistics, planeStatistics = eval_plane_prediction(predict_segmentation, gt_seg,
																	 depth, gt_depth)

			pixel_recall_curve += np.array(pixelStatistics)
			plane_recall_curve += np.array(planeStatistics)

			#print("pixel and plane recall of test image %d ", iter)
			#print(pixel_recall_curve / float(iter+1))
			#print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
			#print("********")

			# visualization convert labels to color image
			# change non planar to zero, so non planar region use the black color
			gt_seg += 1
			gt_seg[gt_seg==21] = 0
			predict_segmentation += 1
			predict_segmentation[predict_segmentation==21] = 0

			gt_seg_image = cv2.resize(np.stack([colors[gt_seg, 0],
												colors[gt_seg, 1],
												colors[gt_seg, 2]], axis=2), (w, h))
			pred_seg = cv2.resize(np.stack([colors[predict_segmentation, 0],
											colors[predict_segmentation, 1],
											colors[predict_segmentation, 2]], axis=2), (w, h))

			# blend image
			blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)
			blend_gt = (gt_seg_image * 0.7 + image * 0.3).astype(np.uint8)

			semantic = cv2.resize((semantic * 255).astype(np.uint8), (w, h))
			semantic = cv2.cvtColor(semantic, cv2.COLOR_GRAY2BGR)

			mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

			depth_diff = np.abs(gt_depth - depth)
			depth_diff[gt_depth == 0.] = 0

			# visualize depth map as PlaneNet
			depth_diff = np.clip(depth_diff / 5 * 255, 0, 255).astype(np.uint8)
			depth_diff = cv2.cvtColor(cv2.resize(depth_diff, (w, h)), cv2.COLOR_GRAY2BGR)

			depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
			depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)

			gt_depth = 255 - np.clip(gt_depth / 5 * 255, 0, 255).astype(np.uint8)
			gt_depth = cv2.cvtColor(cv2.resize(gt_depth, (w, h)), cv2.COLOR_GRAY2BGR)

			image_1 = np.concatenate((image, pred_seg, gt_seg_image), axis=1)
			image_2 = np.concatenate((image, blend_pred, blend_gt), axis=1)
			image_3 = np.concatenate((image, mask, semantic), axis=1)
			image_4 = np.concatenate((depth_diff, depth, gt_depth), axis=1)
			image = np.concatenate((image_1, image_2, image_3, image_4), axis=0)

			#cv2.imshow('image', image)
			#cv2.waitKey(0)
			#cv2.imwrite("%d_segmentation.png"%iter, image)

		print("========================================")
		print("pixel and plane recall of all test image")
		print(pixel_recall_curve / len(data_loader))
		print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
		print("****************************************")


if __name__ == '__main__':
	assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
		'PyTorch>=0.4.0 is required'

	ex.add_config('./configs/config_unet_mean_shift.yaml')
	ex.run_commandline()
