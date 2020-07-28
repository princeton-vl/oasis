import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
from distutils.version import LooseVersion
from sacred import Experiment
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
import glob
from tqdm import tqdm
import argparse

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import k_inv_dot_xy1
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss


ex = Experiment()

transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

    bin_mean_shift = Bin_Mean_Shift()
    instance_parameter_loss = InstanceParameterLoss()

    h, w = 192, 256

    f = open(cfg.dataset_csv, 'r')
    lines = f.readlines()
    f.close()
    lines.pop(0) # header

    with torch.no_grad():
        #for i, image_path in enumerate(tqdm(files)):
        for line in tqdm(lines):
            splits = line.split(',')
            image_path = splits[0]

            image_name = image_path.split('/')[-1]
            image_path = os.path.join(cfg.image_path, image_name)
            #image = cv2.imread(cfg.input_image)
            image = cv2.imread(image_path)
            oh, ow, _ = image.shape
            # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
            image = cv2.resize(image, (w, h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            #
            image = transforms(image)
            image = image.cuda().unsqueeze(0)
            # forward pass
            logit, embedding, _, _, param = network(image)

            prob = torch.sigmoid(logit[0])

            # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
            _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

            # fast mean shift
            segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
            # we thus use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, h, w)
            pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            b = pooling_b.view(-1, h*w).t()
            segmentation = b

            # infer instance depth
            instance_loss, instance_depth, instance_abs_disntace, instance_parameter = \
                instance_parameter_loss(segmentation, sampled_segmentation, sample_param,
                                        torch.ones_like(logit), torch.ones_like(logit), False)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)
            #import pdb; pdb.set_trace()

            # mask out non planar region
            predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
            predict_segmentation = predict_segmentation.reshape(h, w)

            # visualization and evaluation
            image = tensor_to_image(image.cpu()[0])
            mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
            depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            # use per pixel depth for non planar region
            depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)

            # change non planar to zero, so non planar region use the black color
            predict_segmentation += 1
            predict_segmentation[predict_segmentation==21] = 0

            num_ins = predict_segmentation.max()
            seg_path = os.path.join(cfg.output_path, 'seg')
            bboxes = []
            for j in range(1, num_ins + 1):
                m = (predict_segmentation == j).astype(np.float)
                m = m * 255.0
                m = m.astype(np.uint8)
                m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
                x, y = np.where(m >= 127)
                if len(x) == 0:
                    continue
                min_x = np.min(x)
                max_x = np.max(x)
                min_y = np.min(y)
                max_y = np.max(y)
                bbox = [min_x, min_y, max_x, max_y, 1.0]
                bboxes.append(bbox)

            #tqdm.write("{} {}".format(ow, oh))
            #tqdm.write(str(bboxes))
            file_obj = open(os.path.join(cfg.output_path, image_name.split('.')[0] + '.pkl'), 'wb') 
            pickle.dump(bboxes, file_obj)
            file_obj.close()

            pred_seg = cv2.resize(np.stack([colors[predict_segmentation, 0],
                                            colors[predict_segmentation, 1],
                                            colors[predict_segmentation, 2]], axis=2), (w, h))

            # blend image
            blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)

            mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # visualize depth map as PlaneNet
            depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
            depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)

            image = np.concatenate((image, pred_seg, blend_pred, mask, depth), axis=1)

            mask_path = os.path.join(cfg.output_path, 'mask')
            #cv2.imwrite(os.path.join(mask_path, image_name), mask)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/config_unet_mean_shift.yaml')
    ex.run_commandline()
