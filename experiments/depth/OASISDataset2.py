import numpy as np
import random
from utils import save_obj, load_obj

import torch

from torch.utils import data
import cv2
import os
import random

from ReDWebNet import resNet_data_preprocess

def vis_mask(depths):
	out = depths.copy()
	
	out = out - np.min(out)
	out = out / np.max(out) * 255.0
	
	return out

def vis_depth(depths):
	out = depths.copy()
	mask = out > 0
	out[mask] = out[mask] - np.min(out[mask])
	out[mask] = out[mask] / np.max(out[mask]) * 255.0
	
	return out

def draw(img, target, fname):
	img_temp = img.copy()
	
	color_close = (255, 0, 0)	# close is blue
	color_far = (0, 255, 0)		# far is green
	for i in range(target.shape[1]):
		x1 = int(target[1, i]); y1 = int(target[0, i])
		x2 = int(target[3, i]); y2 = int(target[2, i])
		
		cv2.circle(img_temp,(x1, y1),2,color_far,-1)
		cv2.circle(img_temp,(x2, y2),2,color_close,-1)
		cv2.arrowedLine(img_temp, (x2, y2), (x1, y1), (0, 255, 255), 1)
	
	cv2.imwrite(fname, img_temp)
	print( "Done writing to %s" % fname )

class data_augmenter():
	def __init__(self, width, height):
		"""
			Args:
				width and height are only used to determine the 
				output aspect ratio, not the actual output size
		"""
		self.ops = []
		cv2.setNumThreads(0)
		self.width = float(width)
		self.height = float(height)
		
	def add_rotation(self, probability, max_left_rotation=-10, max_right_rotation=10):
		self.ops.append({'type':'rotation', 'probability':probability, 'max_left_rotation': max_left_rotation, 'max_right_rotation':max_right_rotation})
	def add_zoom(self, probability, min_percentage, max_percentage):
		self.ops.append({'type':'zoom', 'probability':probability, 'min_percentage': min_percentage, 'max_percentage': max_percentage})
	def add_flip_left_right(self, probability):
		self.ops.append({'type':'flip_lr', 'probability':probability})
	def add_crop(self, probability, min_percentage=0.5):
		self.ops.append({'type':'crop', 'probability':probability, 'min_percentage':min_percentage})

	def __str__(self):
		out_str = 'Data Augmenter:\n'
		for op in self.ops:
			out_str += '\t'
			for key in op.keys():
				out_str = out_str + str(key) +':'+ str(op[key]) + '\t'
			out_str += '\n'
		return out_str

	def aug(self, img, target):
		"""
			img and target are 2D numpy array with the same size
			img should be H x W x 3
			target should be H x W
		"""
		orig_img = img.copy()
		orig_target = target.copy()
		
		for op in self.ops:
			if random.uniform(0.0, 1.0) <= op['probability']:
				if op['type'] == 'crop':
					percentage = random.uniform(op['min_percentage'], 1.0)
					# print "Cropping.: Percentage = %f" % percentage
					#################### image
					if img.shape[0] <= img.shape[1]:
						dst_h = int(img.shape[0] * percentage)
						dst_w = min(int(dst_h / self.height * self.width), img.shape[1])
					elif img.shape[0] > img.shape[1]:
						dst_w = int(img.shape[1] * percentage)
						dst_h = min(int(dst_w / self.width * self.height), img.shape[0])
					offset_y = random.randint(0, img.shape[0]- dst_h)
					offset_x = random.randint(0, img.shape[1]- dst_w)

					#################### crop
					img = img[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w, :]
					target = target[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w]
					

				elif op['type'] == 'flip_lr':
					# print "Flipping..................."
					#################### image
					img = cv2.flip(img, 1)
					#################### target
					target = cv2.flip(target, 1)

				elif op['type'] == 'zoom':
					# print "Zooming..................."
					percentage = random.uniform(op['min_percentage'], op['max_percentage'])

					#################### image
					img = cv2.resize(img, None, fx = percentage, fy = percentage)

					#################### target
					target = cv2.resize(target, None, fx = percentage, fy = percentage)

				elif op['type'] == 'rotation':
					# print "Rotating..................."
					#################### image
					angle = random.uniform(-op['max_left_rotation'], op['max_right_rotation'])
					rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)


					img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
					target = cv2.warpAffine(target, rotation_matrix, (img.shape[1], img.shape[0]))
					

					# self.draw(img, target, '3_rotation.png')



		return img, target

class OASISDataset(data.Dataset):
	def __init__(self, csv_filename, 
					   height=240, width=320, 
					   b_oppi = False, 
					   b_data_aug = False,
					   b_resnet_prep = False):

		super(OASISDataset, self).__init__()
		print("=====================================================")
		print( "Using ThreeSIW Dataset... With scaling of training focal length!")
		self.get_OASISDataset_meta(csv_filename)

		self.height = height
		self.width = width
		self.n_sample = len(self.img_names)
		self.b_resnet_prep = b_resnet_prep
		self.b_data_aug = False
	
		print( "\t-(width, height): (%d, %d)" % (self.width, self.height))
		print( "\t-%d samples" % (self.n_sample)		)
		print( "\t-Data augmentation:", self.b_data_aug				)
		print( "\t-Resnet data preprocessing:", self.b_resnet_prep)
		print("=====================================================")

	def parse_DIW_csv(self, csv_filename, orig_res):
		# orig_res: height, width
		height, width = orig_res
		y_A_x_A_y_B_x_B_rel = []
		with open(csv_filename, 'r') as f:	
			lines = [line.strip() for line in f]
			if len(lines) == 0:
				return None
			data = np.zeros((5, len(lines)), dtype = np.float32)
			for idx, coords in enumerate(lines):
				# parse coordinates and relation				
				y_A, x_A, y_B, x_B, rel, _a, _b = coords[:-1].split(',')
				
				data[0,idx] = min(float(y_A), height-1)
				data[1,idx] = min(float(x_A), width - 1)
				data[2,idx] = min(float(y_B), height-1)
				data[3,idx] = min(float(x_B), width - 1)
				data[4,idx] = {'<' : -1, '>' : 1}[rel]								

		y_A_x_A_y_B_x_B_rel = np.array(data)

		return y_A_x_A_y_B_x_B_rel

	def create_meta_data(self, csv_filename):
		img_names = []
		metric_depth_names = []
		rel_depth_names = []
		cont_surface_ids = []
		focals = []
		
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
				focals.append(float(splits[1]))
				metric_depth_names.append(splits[4])
				rel_depth_names.append(splits[5])
				
				cont_surface_ids.append(splits[13])
			
		return img_names, metric_depth_names, rel_depth_names, cont_surface_ids, focals


	def get_OASISDataset_meta(self, csv_filename):
		self.img_names, self.metric_depth_names, self.rel_depth_names, self.cont_surface_ids, self.focals = \
																		self.create_meta_data(csv_filename)

	def unpack_depth(self, depth_dict, resolution):
		out_depth = np.zeros(resolution, dtype = np.float32)
		out_depth.fill(-1.0)
		
		min_y = depth_dict['min_y']
		max_y = depth_dict['max_y']
		min_x = depth_dict['min_x']
		max_x = depth_dict['max_x']
		roi_depth = depth_dict['depth']        
		roi_depth[np.isnan(roi_depth)] = -1.0		# there are nan in non-valid regions
		out_depth[min_y:max_y+1, min_x:max_x+1] = roi_depth

		return out_depth

	def __getitem__(self, index):
		# This data reader assumes that the target coordinates are represented 
		# by value in [0, 1.0], i.e., the ratio between the original coordinate
		# and the original image height / image width

		focal = self.focals[index]
		color = cv2.imread(self.img_names[index])
		orig_res = color.shape[:2]				
		if len(color.shape) == 2:
			print("Gray Image: %s" % self.img_names[index])
			color = np.stack([color, color, color], axis = 2)
		
		surface_ids = cv2.imread(self.cont_surface_ids[index])						
		surface_ids = surface_ids[:,:,0]
		
		
		if self.metric_depth_names[index] != '':
			try:
				metric_depth = self.unpack_depth(load_obj(self.metric_depth_names[index]), color.shape[:2])	
				
				# extremely important: there could be area where the depth is invalid,
				# but on surface_ids it is mark with a non zero value
				gt_invalid_mask = metric_depth <= 0
				surface_ids[gt_invalid_mask] = 0			
				
			except:
				print("Error reading %s" % self.metric_depth_names[index])
				metric_depth = None
		else:
			metric_depth = None
		
		
		target = None

		
		#####################################################
		# resize to network input size
		color = cv2.resize(color, (self.width, self.height))
		surface_ids = cv2.resize(surface_ids, (self.width, self.height), 
								 		interpolation=cv2.INTER_NEAREST)
		if metric_depth is not None:
			metric_depth = cv2.resize(metric_depth, (self.width, self.height), 
										interpolation=cv2.INTER_NEAREST)
			surface_ids[metric_depth <= 0] = 0		# extremely important
			
			scaling = (self.width / orig_res[1] + self.height / orig_res[0]) / 2.0
			focal *= scaling
			

			

		
		#####################################################
		# last step of preprocessing
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0			
		surface_ids = np.expand_dims(surface_ids, axis = 0)
		
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)
		
		return color, metric_depth, surface_ids, target, (self.height, self.width), np.array([focal], dtype = np.float32), self.img_names[index]

	def __len__(self):
		return self.n_sample


class OASISDatasetVal(OASISDataset):
	def __init__(self, csv_filename, 
						height=240, width=320, 
						b_oppi = False, 
						b_resnet_prep = False):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("\tValidation version of the OASISDataset")
		print("\t\t-It never perform data augmentation")
		OASISDataset.__init__(self, csv_filename, 
										height = height, width = width, 
										b_oppi = b_oppi, 
										b_data_aug = False, 
										b_resnet_prep = b_resnet_prep)
		

	def __getitem__(self, index):				
		focal = self.focals[index]
		color = cv2.imread(self.img_names[index])
		orig_res = color.shape[:2]		
		if len(color.shape) == 2:
			print("Gray Image: %s" % self.img_names[index])
			color = np.stack([color, color, color], axis = 2)
						
		surface_ids = cv2.imread(self.cont_surface_ids[index])						
		surface_ids = surface_ids[:,:,0]
		


		if self.metric_depth_names[index] != '':
			try:
				metric_depth = self.unpack_depth(load_obj(self.metric_depth_names[index]), color.shape[:2])	
				surface_ids[metric_depth <= 0] = 0		# extremely important			
			except:
				print("Error reading %s" % self.metric_depth_names[index])
				metric_depth = None
		else:
			metric_depth = None
		
		
		if self.rel_depth_names[index] != '':
			try:
				target = self.parse_DIW_csv(self.rel_depth_names[index], orig_res)
			except:
				print("Error reading %s" % self.rel_depth_names[index])
				target = None					
		else:
			target = None


		#####################################################
		# resize to network input size
		color = cv2.resize(color, (self.width, self.height))
				
		if target is not None:
			target = target.astype(np.int64)
		
		#####################################################
		# last step of preprocessing
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0				
		surface_ids = np.expand_dims(surface_ids, axis = 0)

		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)

		return color, metric_depth, surface_ids, target, orig_res, np.array([focal], dtype = np.float32), self.img_names[index]


class OASISDatasetDIWVal(OASISDatasetVal):
	def __init__(self, csv_filename, 
						height=240, width=320, 
						b_oppi = False, 
						b_resnet_prep = False):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("\tValidation version of the OASISDataset: DIW style!")
		print("\t\t-It never perform data augmentation")
		OASISDatasetVal.__init__(self, csv_filename, 
										height = height, width = width, 
										b_oppi = b_oppi, 
										b_resnet_prep = b_resnet_prep)

	def get_OASISDataset_meta(self, csv_filename):
		
		print ("==========  Creating DIW style! OASISDatasetDIWVal !")
		self.img_names, self.metric_depth_names, self.rel_depth_names, self.cont_surface_ids, self.focals = \
																		self.create_meta_data(csv_filename)

	def create_meta_data(self, csv_filename):
		img_names = []
		metric_depth_names = []
		rel_depth_names = []
		cont_surface_ids = []
		focals = []
		
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
				focals.append(float(splits[1]))
				metric_depth_names.append(splits[4])
				rel_depth_names.append(splits[0].replace("image", "DIW_style_rel_depth").replace(".png", ".csv"))
				
				cont_surface_ids.append(splits[13])
			
		return img_names, metric_depth_names, rel_depth_names, cont_surface_ids, focals



def OASIS_collate_fn(batch):
	# color, metric_depth, surface_ids, target.astype(np.int64), (self.height, self.width)
	metric_depth = []
	for b in batch:
		if b[1] is not None:
			metric_depth.append(torch.from_numpy(b[1]))
		else:
			metric_depth.append(torch.FloatTensor())
		# metric_depth.append(torch.FloatTensor())
	
	relative_depth = []
	for b in batch:
		if b[3] is not None:
			relative_depth.append(torch.from_numpy(b[3]))
		else:
			relative_depth.append(torch.LongTensor())
		# relative_depth.append(torch.LongTensor())

	images = [torch.from_numpy(b[0]) for b in batch]
	surface_ids = [torch.from_numpy(b[2]) for b in batch]
	resolution = [b[4] for b in batch]
	focals = [torch.from_numpy(b[5]) for b in batch]
	names = [b[6] for b in batch]
	# DEBUG TODO
	return (torch.stack(images, 0), metric_depth,  torch.stack(surface_ids, 0), relative_depth, resolution, torch.stack(focals, 0), names)


