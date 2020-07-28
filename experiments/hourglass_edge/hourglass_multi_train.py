import os
import cv2
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname

# Customized import.
import time
from models.edgenet import EdgeNet
from OASIS_multi_class_dataset import OASISDataset
from utils_hed import AverageMeter, TBLogger,\
    load_checkpoint, save_checkpoint, load_vgg16_caffe, load_pretrained_caffe, save_obj

import sys
sys.path.append("..")
from upload_util import compress_png

# Parse arguments.
parser = argparse.ArgumentParser(description='HED training.')
# 1. Actions.
parser.add_argument('--test',             default=False,             help='Only test the model.', action='store_true')
parser.add_argument('--val',              default=False,             help='Only validate the model.', action='store_true')
parser.add_argument('--debug',             default=False,             help='Only test the model.', action='store_true')
parser.add_argument('--ignore_classification',   default=False, help='Only inference on boundary.', action='store_true')
parser.add_argument('--ignore_boundary',   default=False, help='Only inference on classification.', action='store_true')
parser.add_argument('--not_gpu_parallel',   default=False, help='Use a single GPU.', action='store_true')

# 2. Counts.
parser.add_argument('--batch_size',       default=4,    type=int,   metavar='N', help='Training batch size.')
parser.add_argument('--train_iter_size',  default=1,   type=int,   metavar='N', help='Training iteration size.')
parser.add_argument('--max_epoch',        default=20,   type=int,   metavar='N', help='Total epochs.')
parser.add_argument('--print_freq',       default=200,  type=int,   metavar='N', help='Print frequency.')
parser.add_argument('--default_epoch',    default=0,  type=int,   metavar='N', help='Default Epoch.')

# 3. Optimizer settings.
parser.add_argument('--lr',               default=2e-4, type=float, metavar='F', help='Initial learning rate.')
parser.add_argument('--lr_stepsize',      default=1e4,  type=int,   metavar='N', help='Learning rate step size.')
parser.add_argument('--lr_gamma',         default=1.0,  type=float, metavar='F', help='Learning rate decay (gamma).')
parser.add_argument('--alpha',     default=1, type=float, metavar='F', help='Alpha.')
parser.add_argument('--beta',     default=1, type=float, metavar='F', help='Beta.')

# 4. Files and folders.
parser.add_argument('--vgg16_caffe',      default='',                help='Resume VGG-16 Caffe parameters.')
parser.add_argument('--checkpoint',       default='',                help='Resume the checkpoint.')
parser.add_argument('--caffe_model',      default='',                help='Resume HED Caffe model.')
parser.add_argument('--dataset_name',     default='OASIS_occ_fold_trainval',   help='OASIS_occ_fold_trainval')

# 5. Network settings.
parser.add_argument('--num_output_channel',  default=2, type=int, metavar='N', help='Hourglass number of output channel.')
parser.add_argument('--nstack',  default=1, type=int, metavar='N', help='Hourglass number of stack.')

# 6. Others.
parser.add_argument('--is_data_aug',   default=False,      help='Enable data augumentation.', action='store_true')
parser.add_argument('--cpu',              default=False,             help='Enable CPU mode.', action='store_true')
parser.add_argument('--exp_name',         default='debug_multi_class',           help='The name of this experiment.')
parser.add_argument('--train_split',      default='train',           help='data/OASIS_occ_fold_trainval/XXX.txt')

# 7. Upload
parser.add_argument('--upload',   default=False,      help='Upload mode.', action='store_true')


args = parser.parse_args()
assert(not(args.ignore_boundary and args.ignore_classification))

# Set device.
device = torch.device('cpu' if args.cpu else 'cuda')



from PIL import Image
import numpy as np

def compress_save(data, out_filename):
    '''
        Use png encoding to reduce the size of the saved data

        data: single precision 32 bit 2D/3D numpy array
        out_filename: png filename
    '''
    if len(data.shape) == 3:
        height, width, channel = data.shape
    else:
        height, width = data.shape
        channel = 1

    data_float = data.astype(np.float32)
    data_byte = data_float.tobytes()
    data_uint16 = np.frombuffer(data_byte, dtype=np.uint16)
    data_uint16 = data_uint16.reshape((height, width * channel * 2))
    
    img1 = Image.fromarray(data_uint16)
    img1.save(out_filename)

def main():
    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    output_dir = join('./exps', args.exp_name)
    if not isdir(output_dir):
        os.makedirs(output_dir)
    save_obj(vars(args), os.path.join(output_dir, 'args.pkl'))

    # Set logger.
    tb_logger = TBLogger(folder = output_dir)
    # tb_logger.create_scalar(name = 'lr')
    # tb_logger.create_scalar(name = 'batch_loss')
    # tb_logger.create_scalar(name = 'avg_batch_loss')


    ################################################
    # II. Datasets.
    ################################################
    _dataset_dir = {
                    'OASIS_occ_fold_test':     './data/OASIS_occ_fold_test',
                    'OASIS_occ_fold_trainval': './data/OASIS_occ_fold_trainval'
                   }

    # Datasets and dataloaders.
    if args.test:
        assert(args.dataset_name == 'OASIS_occ_fold_test')
        test_dataset  = OASISDataset(split='test', dataset_dir=_dataset_dir[args.dataset_name])
        test_loader   = DataLoader(test_dataset,  batch_size=1,
                               num_workers=4, drop_last=False, shuffle=False)
    elif args.val:
        assert(args.dataset_name == 'OASIS_occ_fold_trainval')
        val_dataset   = OASISDataset(split='val', dataset_dir=_dataset_dir[args.dataset_name])
        val_loader    = DataLoader(val_dataset,  batch_size=1,
                               num_workers=4, drop_last=False, shuffle=False)
    else:
        assert(args.dataset_name == 'OASIS_occ_fold_trainval')
        train_dataset = OASISDataset(split=args.train_split, dataset_dir=_dataset_dir[args.dataset_name], is_data_aug=args.is_data_aug)
        val_dataset   = OASISDataset(split='val', dataset_dir=_dataset_dir[args.dataset_name])

        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               num_workers=4, drop_last=True, shuffle=True)
        val_loader    = DataLoader(val_dataset,  batch_size=1,
                               num_workers=4, drop_last=False, shuffle=False)
    ################################################
    # III. Network and optimizer.
    ################################################
    # Specify the configuration for Hourglass
    configs= {
        'inference': {
            'nstack': args.nstack,
            'inp_dim': 64,
            'oup_dim': args.num_output_channel,
            'num_parts': 17,
            'increase': 128,
            'keys': ['imgs']
        },
    }
    # Create the network in GPU.
    config = configs['inference']
    if args.not_gpu_parallel:
        net = EdgeNet(**config)
    else:
        net = nn.DataParallel(EdgeNet(**config))
    net.to(device)

    # Initialize the weights for EdgeNet model. (TODO)
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            torch.nn.init.xavier_uniform_(m.weight)
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()
    net.apply(weights_init)

    # Optimizer settings.
    # Create optimizer.
    #print("Parameters to be optimized: {}".format(len(list(net.named_parameters()))))
    opt = torch.optim.Adam(net.parameters(), args.lr)
    # Learning rate scheduler.
    lr_schd = lr_scheduler.StepLR(opt, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)  # Omit the returned values.

    ################################################
    # V. Training / testing.
    ################################################
    if args.test:
        # Only test.
        epoch_name = os.path.basename(args.checkpoint)
        epoch_name = epoch_name.replace("-checkpoint.pt", "")

        test(test_loader, net, save_dir=join(output_dir, '{}-test'.format(epoch_name)))

    elif args.val:
        # Only val.
        epoch_name = os.path.basename(args.checkpoint)
        epoch_name = epoch_name.replace("-checkpoint.pt", "")

        test(val_loader, net, save_dir=join(output_dir, '{}-val'.format(epoch_name)))            
    else:
        # Train.
        train_epoch_losses = []
        # Train main loop
        for epoch in range(args.max_epoch):
            epoch = epoch + args.default_epoch
            # Initial test.
            if epoch == 0: #not args.debug:
                print('Initial val...')
                test(val_loader, net, save_dir=join(output_dir, 'initial-val'))
                
            # Epoch training and test.
            train_epoch_loss = \
                train(train_loader, net, opt, lr_schd, epoch, save_dir=join(output_dir, 'epoch-{}-train'.format(epoch)), tb_logger = tb_logger)
            
            # Validation
            if epoch % 2 == 0:
                test(val_loader, net, save_dir=join(output_dir, 'epoch-{}-val'.format(epoch)))
                
            # # Write log.
            # tb_logger.write_to_file()
            # Save checkpoint.
            save_checkpoint(state={'net': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                            path=os.path.join(output_dir, 'epoch-{}-checkpoint.pt'.format(epoch)))
            # Collect losses.
            train_epoch_losses.append(train_epoch_loss)


def train(train_loader, net, opt, lr_schd, epoch, save_dir, tb_logger):
    """ Training procedure. """
    # Create the directory.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # Switch to train mode and clear the gradient.
    net.train()
    opt.zero_grad()
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()
    batch_loss_b_meter = AverageMeter()
    batch_loss_c_meter = AverageMeter()

    # Note: The counter is used here to record number of batches in current training iteration has been processed.
    #       It aims to have large training iteration number even if GPU memory is not enough. However, such trick
    #       can be used because batch normalization is not used in the network architecture.
    counter = 0
    for batch_index, (images, edges, ROIs, _, _) in enumerate(tqdm(train_loader)):
        # Adjust learning rate and modify counter following Caffe's way.
        if counter == 0:
            lr_schd.step()  # Step at the beginning of the iteration.
        counter += 1

        # Get images and edges from current batch.
        images, edges, ROIs = images.to(device), edges.to(device), ROIs.to(device)

        # Generate predictions.
        preds_list = net(images)

        # Calculate the loss of current batch (sum of all scales and fused).
        # Note: Here we mimic the "iteration" in official repository: iter_size batches will be considered together
        #       to perform one gradient update. To achieve the goal, we calculate the equivalent iteration loss
        #       eqv_iter_loss of current batch and generate the gradient. Then, instead of updating the weights,
        #       we continue to calculate eqv_iter_loss and add the newly generated gradient to current gradient.
        #       After iter_size batches, we will update the weights using the accumulated gradients and then zero
        #       the gradients.
        # Reference:
        #   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L230
        #   https://www.zhihu.com/question/37270367
        losses = [cross_entropy_loss(preds, edges, ROIs, args.alpha, args.beta) for preds in preds_list]
        batch_loss = sum([losses[i][0] for i in range(len(losses))])
        batch_loss_b = sum([losses[i][1] for i in range(len(losses))])
        batch_loss_c = sum([losses[i][2] for i in range(len(losses))])
        eqv_iter_loss = batch_loss / args.train_iter_size

        # Generate the gradient and accumulate (using equivalent average loss).
        eqv_iter_loss.backward()
        if counter == args.train_iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0  # Reset the counter.
        # Record loss.
        batch_loss_meter.update(batch_loss.item())
        batch_loss_b_meter.update(batch_loss_b.item())
        batch_loss_c_meter.update(batch_loss_c.item())
        # Log and save intermediate images.
        if batch_index % args.print_freq == args.print_freq - 1:

            #########################################################################
            # Log.
            print(('Training epoch:{}/{}, batch:{}/{} current iter:{}, ' +
                   'epoch average loss:{},' +
                   'epoch average bd_batch_loss:{}, epoch average cs_batch_loss:{},' +
                   'learning rate list:{}.').format(
                   epoch, args.max_epoch, batch_index, len(train_loader), lr_schd.last_epoch,
                   batch_loss_meter.avg, batch_loss_b_meter.avg, batch_loss_c_meter.avg,
                   lr_schd.get_lr()))
            tb_logger.add_value('lr', lr_schd.get_lr()[0], step=lr_schd.last_epoch)
            tb_logger.add_value('batch_loss', batch_loss_meter.val, step=lr_schd.last_epoch)
            tb_logger.add_value('avg_batch_loss', batch_loss_meter.avg, step=lr_schd.last_epoch)
            # tb_logger.write_to_file()

            #########################################################################
            # Generate intermediate images.
            preds_list_and_edges = preds_list + [edges]
            _, _, h, w = preds_list_and_edges[0].shape
            b_interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            c_interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            for i in range(len(preds_list_and_edges)):
                # Only fetch the first image in the batch.
                if i != (len(preds_list_and_edges)-1):
                    b_interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
                    c_interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 1, :, :]
                else:
                    b_interm_images[i, 0, :, :] =  torch.clamp(preds_list_and_edges[i][0, 0, :, :] + preds_list_and_edges[i][0, 1, :, :], min=0, max=1)
                    c_interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            # Save the images.
            torchvision.utils.save_image(b_interm_images, join(save_dir, 'batch-{}-1st-boundary_image.png'.format(batch_index))) # boundary
            torchvision.utils.save_image(c_interm_images, join(save_dir, 'batch-{}-1st-class_image.png'.format(batch_index))) # classification
    # Return the epoch average batch_loss.
    return batch_loss_meter.avg


# Potentially fix. Three formulation is possible.
# 1. Hard-labels (argmax)
# 2. Soft-labels (a raw probability) masked by argmax
# 3. Soft-labels
def test(test_loader, net, save_dir, is_train_val = False):
    """ Test procedure. """
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    if args.dataset_name in ["OASIS_occ_fold_trainval", "OASIS_occ_fold_test"]:
       boundary_save_dir = join(save_dir, 'boundary')
       if not isdir(boundary_save_dir):
           os.makedirs(boundary_save_dir)
       boundary_save_png_dir = join(boundary_save_dir, 'png')
       if not isdir(boundary_save_png_dir):
           os.makedirs(boundary_save_png_dir)
       boundary_save_mat_dir = join(boundary_save_dir, 'mat')
       if not isdir(boundary_save_mat_dir):
           os.makedirs(boundary_save_mat_dir)
       if args.ignore_classification:
          class_occ_save_dir = join(save_dir, 'class_occ')
          if not isdir(class_occ_save_dir):
              os.makedirs(class_occ_save_dir)
          class_occ_save_png_dir = join(class_occ_save_dir, 'png')
          if not isdir(class_occ_save_png_dir):
              os.makedirs(class_occ_save_png_dir)
          class_occ_save_mat_dir = join(class_occ_save_dir, 'mat')
          if not isdir(class_occ_save_mat_dir):
              os.makedirs(class_occ_save_mat_dir)
          class_fold_save_dir = join(save_dir, 'class_fold')
          if not isdir(class_fold_save_dir):
              os.makedirs(class_fold_save_dir)
          class_fold_save_png_dir = join(class_fold_save_dir, 'png')
          if not isdir(class_fold_save_png_dir):
              os.makedirs(class_fold_save_png_dir)
          class_fold_save_mat_dir = join(class_fold_save_dir, 'mat')
          if not isdir(class_fold_save_mat_dir):
              os.makedirs(class_fold_save_mat_dir)
       else:
           class_save_dir = join(save_dir, 'class')
           if not isdir(class_save_dir):
               os.makedirs(class_save_dir)
           class_save_png_dir = join(class_save_dir, 'png')
           if not isdir(class_save_png_dir):
               os.makedirs(class_save_png_dir)
           class_save_mat_dir = join(class_save_dir, 'mat')
           if not isdir(class_save_mat_dir):
               os.makedirs(class_save_mat_dir)
    # Switch to evaluation mode.
    net.eval()
    max_iter = 1000000
    if save_dir.find("val") >= 0:
        max_iter = 2000

    # Generate predictions and save.
    for batch_index, (images, edge, _, h, w) in enumerate(tqdm(test_loader)):
        if is_train_val and batch_index > 100:
            break
        if batch_index > max_iter:
            break
            
        images = images.cuda()
        torch.cuda.empty_cache()
        if h > 1024 or w > 1024:
            print("WARNING: {} too large.".format(test_loader.dataset.images_name[batch_index]))
            continue
        preds_list = net(images)
        fuse_tensor = preds_list[-1].detach()[0,:,:,:]
        fuse = fuse_tensor.cpu().numpy() # Shape: [h, w].
        name = test_loader.dataset.images_name[batch_index]
        if args.dataset_name in ["OASIS_occ_fold_trainval", "OASIS_occ_fold_test"]:
            # occlusion / fold / neither
            boundary_img = fuse[0,:,:].astype(np.float32)
            class_img = fuse[1,:,:].astype(np.float32)

            if args.upload:
                # upload to the eval server
                compress_png(data_float32 = boundary_img, out_pngname = join(boundary_save_mat_dir, '{}.png'.format(name)))
                compress_png(data_float32 = class_img, out_pngname = join(class_save_mat_dir, '{}.png'.format(name)))

            else:
                boundary_img = cv2.resize(boundary_img, (w, h))
                boundary_img = np.clip(boundary_img, 0, 1)
                class_img = cv2.resize(class_img, (w, h))
                class_img = np.clip(class_img, 0, 1)
                
                #  boundary image
                sio.savemat(join(boundary_save_mat_dir, '{}.mat'.format(name)), {'result': boundary_img})
                Image.fromarray((boundary_img * 255).astype(np.uint8)).save(join(boundary_save_png_dir, '{}.png'.format(name)))

                # class prediction (fake)
                sio.savemat(join(class_save_mat_dir, '{}.mat'.format(name)), {'result': class_img})
                Image.fromarray((class_img * 255).astype(np.uint8)).save(join(class_save_png_dir, '{}.png'.format(name)))



def cross_entropy_loss(preds, edges, ROIs, alpha = 1, beta=1, default_weight = 1e-3):
    """ Calculate sum of weighted cross entropy loss. """
    b, c, h, w = preds.shape # bx2xhx2
    assert(c == 2) # hard-code only occlusion v.s., fold case

    # Boundary loss
    boundary = edges[:,0,:,:] + edges[:,1,:,:] # edges, 0: occlusion, 1: fold, 2:background
    boundary = boundary.unsqueeze(1)
    mask = (boundary > 0.5).float()
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
    num_neg = torch.sum(ROIs, dim=[1, 2, 3]).float() - num_pos
    b_weight = torch.zeros_like(boundary)
    for i in range(b):
        if num_pos[i] == 0:
            b_weight[i,:,:,:] = torch.ones_like(mask[i,:,:,:]) * default_weight
        else:
            b_weight[i,:,:,:][boundary[i,:,:,:] > 0.5]  = num_neg[i] / (num_pos[i] + num_neg[i])
            b_weight[i,:,:,:][boundary[i,:,:,:] <= 0.5] = num_pos[i] / (num_pos[i] + num_neg[i])
        b_weight[i,:,:,:][ROIs[i,:,:,:] == 0] = 0
    b_preds = preds[:,0,:,:].unsqueeze(1)
    b_losses = torch.nn.functional.binary_cross_entropy(b_preds.float(), boundary.float(), weight=b_weight, reduction='none')

    # Categpry classification loss
    # Positive: occlusion, Negative: fold
    category = edges[:,0,:,:]
    category = category.unsqueeze(1)
    c_weight = torch.zeros_like(category)
    num_pos = torch.sum(edges[:,0,:,:] > 0.5, dim=[1, 2]).float()  # Shape: [b,].
    num_neg = torch.sum(edges[:,1,:,:] > 0.5, dim=[1, 2]).float()
    for i in range(b):
        c_weight[i,:,:,:][edges[i,0,:,:].unsqueeze(0) > 0.5]  = num_neg[i] / (num_pos[i] + num_neg[i])
        c_weight[i,:,:,:][edges[i,1,:,:].unsqueeze(0) > 0.5] = num_pos[i] / (num_pos[i] + num_neg[i])
    c_preds = preds[:,1,:,:].unsqueeze(1)
    c_losses = torch.nn.functional.binary_cross_entropy(c_preds.float(), category.float(), weight=c_weight, reduction='none')

    if args.ignore_classification:
        alpha = 0
    if args.ignore_boundary:
        beta = 0

    losses = alpha * c_losses + beta * b_losses
    loss   = torch.sum(losses) / b
    b_loss   = torch.sum(beta * b_losses) / b
    c_loss   = torch.sum(alpha * c_losses) / b
    return loss, b_loss, c_loss

if __name__ == '__main__':
    main()

