import torch
from torch.utils import data
import os
import numpy as np
import cv2
from PIL import Image


class SurfaceDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None):
        assert subset in ['train', 'train_25', 'train_50', 'val']
        print("======================")
        print("Surface dataset, split = %s" % subset)
        self.subset = subset
        self.transform = transform
        self.root_dir = os.path.join(root_dir, subset)
        self.txt_file = os.path.join(root_dir, subset + '.txt')

        self.data_list = [line.strip() for line in open(self.txt_file, 'r').readlines()]
        self.precompute_K_inv_dot_xy_1()
        print("# samples = %d" % len(self.data_list))

    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20

        plane = plane[:plane_nums]

        tmp = plane[:, 1].copy()
        plane[:, 1] = -plane[:, 2]
        plane[:, 2] = tmp

        # convert plane from n * d to n / d
        plane_d = np.linalg.norm(plane, axis=1)
        # normalize
        plane /= plane_d.reshape(-1, 1)
        # n / d
        plane /= plane_d.reshape(-1, 1)

        h, w = segmentation.shape
        plane_parameters = np.ones((3, h, w))
        for i in range(h):
            for j in range(w):
                d = segmentation[i, j]
                if d >= 20: continue
                try:
                    plane_parameters[:, i, j] = plane[d, :]
                except:
                    import pdb; pdb.set_trace()
                    pass

        if np.isnan(plane_parameters).sum() != 0:
            import pdb; pdb.set_trace()

        # plane_instance parameter, padding zero to fix size
        plane_instance_parameter = np.concatenate((plane, np.zeros((20-plane.shape[0], 3))), axis=0)
        return plane_parameters, valid_region, plane_instance_parameter

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = np.dot(self.K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        # precompute to speed up processing
        self.K_inv_dot_xy_1 = K_inv_dot_xy_1

    def plane2depth(self, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):
        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        # replace non planer region depth using sensor depth map
        depth_map[segmentation == 20] = gt_depth[segmentation == 20]
        return depth_map

    def __getitem__(self, index):
        #if self.subset == 'train':
        #    data_path = self.data_list[index]
        #else:
        #    data_path = str(index) + '.npz'
        data_path = self.data_list[index]
        data_path = os.path.join(self.root_dir, data_path)
        data = np.load(data_path)

        image = data['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        plane = data['plane']
        num_planes = data['num_planes'][0]

        gt_segmentation = data['segmentation']
        gt_segmentation = gt_segmentation.reshape((192, 256))
        segmentation = np.zeros([21, 192, 256], dtype=np.uint8)

        _, h, w = segmentation.shape
        for i in range(num_planes+1):
            # deal with backgroud
            if i == num_planes:
                seg = gt_segmentation == 20
            else:
                seg = gt_segmentation == i

            segmentation[i, :, :] = seg.reshape(h, w)

        # surface plane parameters
        plane_parameters, valid_region, plane_instance_parameter = self.get_plane_parameters(plane, num_planes, gt_segmentation)

        # since some depth is missing, we use plane to recover those depth following PlaneNet
        gt_depth = data['depth'].reshape(192, 256)

        #depth = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)
        depth = np.zeros((1, 192, 256))

        # decide mask
        depth_mask = gt_depth >= 0.0
        depth[0] = gt_depth

        # normalize depth
        depth /= 100.0

        valid_region = np.logical_and(depth_mask, valid_region)
        #if valid_region.sum() == 0:
        #    return self.__getitem__(0)
        #print(str(index) + " " + str(valid_region.sum()))

        sample = {
            'image' : image,
            'num_planes': num_planes,
            'roi': depth_mask,
            'instance' : torch.ByteTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation),
            'depth': torch.FloatTensor(depth),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter),
            'img_name':data_path,
        }

        return sample

    def __len__(self):
        return len(self.data_list)
