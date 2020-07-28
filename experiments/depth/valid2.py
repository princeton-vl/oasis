import argparse
import os
import math
import torch
import numpy as np
import cv2

from torch.utils import data
from torch.autograd import Variable


# code for scale invaraint metric depth evaluation
import sys
sys.path.append("../../eval/metric_depth/")
from surface_wise_3D_MSE import obtain_surface_ids, surface_wise_3D_MSE_no_order

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


#### for debug
def classify(z_A, z_B, ground_truth, thresh):
    n_point = z_A.shape[0]

    eq_correct_count = 0.0
    not_eq_correct_count = 0.0
    eq_count = 0.0
    not_eq_count = 0.0

    for i in range(n_point):
        z_A_z_A = z_A[i] - z_B[i]

        _classify_res = 1.0
        if z_A_z_A > thresh:
            _classify_res = 1
        elif z_A_z_A < -thresh:
            _classify_res = -1
        elif z_A_z_A <= thresh and z_A_z_A >= -thresh:
            _classify_res = 0

        if _classify_res == ground_truth[i]:
            if ground_truth[i] == 0:
                eq_correct_count += 1
            else:
                not_eq_correct_count += 1
        
        if ground_truth[i] == 0:
            eq_count += 1
        else:
            not_eq_count += 1

    print ('classify')
    print ('thresh', thresh)
    print ('eq_correct_count', eq_correct_count)
    print ('not_eq_correct_count', not_eq_correct_count)
    print ('eq_count', eq_count)
    print ('not_eq_count', not_eq_count)


def evalute_correct_rate_one_img(depth, target, threshes, target_res = None):
    assert len(target) == 1
    # Target is a list of cpu tensor. There is only one element in the list. This element is a 5 x n Tensor.
    # Depth is a cpu tensor. With 4 dimensions.

    depth = depth[0,0,...]
    target = target[0]

    if target_res is not None:
        #print ("Resizing network output from (%d, %d) to (%d, %d)" % (depth.shape[0], depth.shape[1], target_res[0], target_res[1]))
        depth = resize_tensor(depth, target_res)

    y_A = target[0,:]
    x_A = target[1,:]
    y_B = target[2,:]
    x_B = target[3,:]
    gt_r = target[4,:].float()




    z_A_arr = depth.index_select(1, x_A).gather(0, y_A.view(1,-1))
    z_B_arr = depth.index_select(1, x_B).gather(0, y_B.view(1,-1))
    z_A_z_B = z_A_arr - z_B_arr

    n_pair = target.shape[1]
    eq_count = torch.sum(torch.eq(gt_r, 0.0))
    not_eq_count = torch.sum(torch.eq(gt_r, 1.0)) + torch.sum(torch.eq(gt_r, -1.0))
    assert eq_count + not_eq_count == n_pair

    
    report = {'eq_count':eq_count, 'not_eq_count':not_eq_count} 
    for thresh in threshes:
        _gt_res = torch.gt(z_A_z_B, thresh)
        _lt_res = torch.lt(z_A_z_B, -thresh)

        est_r = _gt_res.float() + _lt_res.float() * -1
        not_eq_correct_count = torch.sum(torch.eq(est_r * gt_r, 1.0))

        eq_mask = 1.0 - torch.abs(gt_r)
        eq_correct_count = torch.sum(eq_mask * (1.0 - torch.abs(est_r)))
    
        report[thresh] = {}
        report[thresh]['not_eq_correct_count'] = not_eq_correct_count
        report[thresh]['eq_correct_count'] = eq_correct_count

    return report


def resize_tensor(depth_tensor, target_res):
    # target_res: a tuple, (height, width)
    if depth_tensor.shape[0] == target_res[0] and depth_tensor.shape[1] == target_res[1]:
        return depth_tensor
    else:
        depth = depth_tensor.numpy()
        depth = cv2.resize(depth, (target_res[1], target_res[0]))	#cv2.resize(src, (width, height))
        depth = torch.from_numpy(depth)
        return depth


def get_pred_rel_depth(rel_depth_pairs, pred):
    '''
        Convert relative depth pairs into surface_wise_3D_MSE format.
        Plus, get the relative depth relation according to the **PRED**
        See eval/surface_wise_3D_MSE.py for more details.
        Input:
            rel_depth_pairs: the 5xN tensor from ThreeSIWDataset.py
            pred: The HxW **predicted** depth 
    '''
    n_sample = rel_depth_pairs.shape[1]
    xA_yA_xB_yBs = []
    
    for i in range(n_sample):
        # rel = rel_depth_pairs[4, i]
        y_A = rel_depth_pairs[0, i]
        x_A = rel_depth_pairs[1, i]
        y_B = rel_depth_pairs[2, i]
        x_B = rel_depth_pairs[3, i]
        
        d_A = pred[y_A, x_A]
        d_B = pred[y_B, x_B]
        
        if d_A > d_B:
            # '>', d_A > d_B, see parse_DIW_csv of ThreeSIWDataset
            xA_yA_xB_yBs.append( (x_A, y_A, x_B, y_B) )
        else:
            xA_yA_xB_yBs.append( (x_B, y_B, x_A, y_A) )
    
    return xA_yA_xB_yBs

def metric_gt_no_order(gt_np, orig_size_pred_np, focal_pred, focal_gt, record):
    '''
        normalize the gt depth, scale the prediction
    '''
    surface_ids = obtain_surface_ids(gt_np)

    _lsiv_metric = surface_wise_3D_MSE_no_order(pred = gt_np,
                                                gt = orig_size_pred_np,                                                
                                                focal_pred = focal_gt, 
                                                focal_gt = focal_pred,
                                                surface_ids = surface_ids,                                                 
                                                )

    print("Normalize GT no order!\t no order lsiv_metric: %g" % _lsiv_metric['loss_sum'])
    # print("\t meta:", _meta)
    record['n_pixel'] += _lsiv_metric['n_pixel']	
    record['loss_sum'] += _lsiv_metric['loss_sum']

    return record

def convert_rel_depth(rel_depth_pairs):
    '''
        Convert relative depth pairs into surface_wise_3D_MSE format.
        See eval/surface_wise_3D_MSE.py for more details.
        Input:
            rel_depth_pairs: the 5xN tensor from ThreeSIWDataset.py
    '''
    n_sample = rel_depth_pairs.shape[1]
    xA_yA_xB_yBs = []
    
    for i in range(n_sample):
        rel = rel_depth_pairs[4, i]
        y_A = rel_depth_pairs[0, i]
        x_A = rel_depth_pairs[1, i]
        y_B = rel_depth_pairs[2, i]
        x_B = rel_depth_pairs[3, i]
        if rel == 1: # '>', d_A > d_B, see parse_DIW_csv of ThreeSIWDataset
            xA_yA_xB_yBs.append( (x_A, y_A, x_B, y_B) )
        elif rel == -1:
            xA_yA_xB_yBs.append( (x_B, y_B, x_A, y_A) )
    
    return xA_yA_xB_yBs

def valid(model, data_loader, criterion, 
            max_iter=500, verbal=False, 
            b_vis_depth=False, 
            in_thresh = None,
            b_eval_rel_depth_only = False):
    print("#######################################################################")
    print("Evaluating... Scale predicted focal lenght!")
    print("\tb_eval_rel_depth_only = %s" % b_eval_rel_depth_only)
    print("#######################################################################\n\n\n")
    
    threshes = []
    if in_thresh is None:
        for i in range(140):
            threshes.append(0.1 + 0.01 * i)
    else:
        if isinstance(in_thresh, list):
            threshes = in_thresh
        else:
            threshes = [in_thresh]

    assert not model.training

    iter = 0 
    reports = []
    metric_error_pred = {'n_pixel': 0.0, 'lsiv_metrics':[], 'training_losses':[]}
    metric_error_gt = {'n_pixel': 0.0, 'lsiv_metrics':[], 'training_losses':[]}
    metric_error_gt_no_order = {'n_pixel': 0.0, 'loss_sum': 0.0}
    for step, (inputs, metric_depth, _, target, target_res, focal_gt, names) in enumerate(data_loader):
        iter += 1
        
        target_res = target_res[0]		# (height, width)
        print(iter)
        
        if iter > max_iter:
            break
        input_var = inputs.to(_device)
        output_var, focal_pred_var = model(input_var)

        ########### for stack hourglass training
        if isinstance(output_var, list):
            output_var = output_var[1]

        pred_np = output_var.cpu().detach().numpy()[0,0,:,:]
        focal_pred = focal_pred_var.cpu().detach().item()
        focal_gt = focal_gt.item()
        
        if target[0].nelement() != 0:
            target_np = target[0].numpy()
        else:
            target_np = None
        if metric_depth[0].nelement() != 0:
            gt_np   = metric_depth[0].numpy()
        else:
            gt_np = None


        if b_vis_depth:
            vis_depth(pred_np, input_var.cpu().detach().numpy(), step, target_res)


        # relative depth
        if target_np is not None:
            report = evalute_correct_rate_one_img(output_var.data.cpu(), target, threshes, target_res)
            reports.append(report)
        

        if not b_eval_rel_depth_only:
            # metric depth
            if gt_np is not None:

                orig_size_pred_np = cv2.resize(pred_np, (target_res[1], target_res[0]))

                ################################
                scaling = (target_res[0] / pred_np.shape[0] + target_res[1] / pred_np.shape[1]) / 2.0                
                focal_pred *= scaling
                ################################


                ##################################################################################
                # metric 3: normalize the gt depth, scale the pred, but no depth order considered
                metric_error_gt_no_order = metric_gt_no_order(gt_np, orig_size_pred_np, focal_pred, focal_gt, metric_error_gt_no_order)
                ################################################################################## 



        # this None assignment is necessary to keep the gpu memory clean
        input_var = None
        output_var = None
        inputs = None
        target = None
        target_res = None


    print ("%d samples are evaluated.\n" % (iter))
    if not b_eval_rel_depth_only:
        metric_error_gt_no_order["LSIV"] = metric_error_gt_no_order['loss_sum'] / metric_error_gt_no_order['n_pixel']
        print ("metric_error_gt_no_order LSIV Metric: %g" % metric_error_gt_no_order["LSIV"])
    
    print ("Thresh\tWKDR\tWKDR_neq\tWKDR_eq")	
    highest_correct_rate = 0	
    relative_error = {"thresh":-1.0, "WKDR":100, "WKDR_neq": 100, "WKDR_eq":100}
    for thresh in threshes:
        eq_correct_count = torch.tensor(0.0)
        not_eq_correct_count = torch.tensor(0.0)
        eq_count = torch.tensor(1e-8)
        not_eq_count = torch.tensor(1e-8)

        for report in reports:			
            eq_correct_count += report[thresh]['eq_correct_count'].float()
            not_eq_correct_count += report[thresh]['not_eq_correct_count'].float()
            eq_count += report['eq_count'].float()
            not_eq_count += report['not_eq_count'].float()

        # Convert tensor to Python primitive
        eq_correct_count = eq_correct_count.item()
        not_eq_correct_count = not_eq_correct_count.item()
        eq_count = eq_count.item()
        not_eq_count = not_eq_count.item()


        correct_rate_neq = not_eq_correct_count / not_eq_count
        if eq_count == 0:
            correct_rate_eq = 0.0
        else:
            correct_rate_eq = eq_correct_count / eq_count
        correct_rate = (not_eq_correct_count + eq_correct_count) / (not_eq_count + eq_count)
        

        if len(threshes) == 1:
            relative_error = {"thresh":thresh, "WKDR":(1.0-correct_rate) * 100, "WKDR_neq": (1.0-correct_rate_neq) * 100, "WKDR_eq":(1.0-correct_rate_eq) * 100}

        if min(correct_rate_neq, correct_rate_eq) > highest_correct_rate:
            highest_correct_rate = min(correct_rate_neq, correct_rate_eq)
            relative_error = {"thresh":thresh, "WKDR":(1.0-correct_rate) * 100, "WKDR_neq": (1.0-correct_rate_neq) * 100, "WKDR_eq":(1.0-correct_rate_eq) * 100}
        
        if verbal:
            print ("%.2f\t%.2f%%\t%.2f%%\t%.2f%%" % (thresh, (1.0-correct_rate) * 100, (1.0-correct_rate_neq) * 100, (1.0-correct_rate_eq) * 100))

    
    print ("Best:")
    print ("%.2f\t%.2f%%\t%.2f%%\t%.2f%%\n" % (relative_error['thresh'], relative_error['WKDR'], relative_error['WKDR_neq'], relative_error['WKDR_eq']))

    return relative_error, metric_error_pred, metric_error_gt, metric_error_gt_no_order



def valid_known(pred_folder, 
                data_loader, 
                max_iter = 100000000000,
                b_plane = False,				
                fixed_focal_len=None,	
                verbal=False, 			
                b_vis_depth=False,
                b_eval_rel_depth_only = False,
                b_keep_neg = False):
    print("#######################################################################")
    print("Evaluating Pre-predicted Results...")
    print("\tpred_folder = %s" % pred_folder)
    print("\tfixed_focal_len =", fixed_focal_len)
    print("\tb_plane = %s" % b_plane)
    print("\tb_eval_rel_depth_only = %s" % b_eval_rel_depth_only)
    print("#######################################################################\n\n\n")

    iter = 0 
    metric_error_gt_no_order = {'n_pixel': 0.0, 'loss_sum':0}
    relative_error = {"thresh":0.0, "WKDR":0}

    not_eq_correct_count = 0.0
    not_eq_count = 0.0

    for step, (inputs, metric_depth, _, target, target_res, focal_gt, names) in enumerate(data_loader):
        iter += 1
        if iter > max_iter:
            break
        #################### read in everything
        target_res = target_res[0]		# (height, width)
        focal_gt = focal_gt.item()		
        if target[0].nelement() != 0:
            target_np = target[0].numpy()
        else:
            target_np = None
        if metric_depth[0].nelement() != 0:
            gt_np = metric_depth[0].numpy()
        else:
            gt_np = None
            print(names[0] + ' does not have gt metric depth.')

        #################### read in the predictions
        if b_plane:
            if verbal:
                print(iter, names[0], "Planar")
            pred_np = np.ones(target_res)		
            assert(fixed_focal_len is not None)
            pred_focal = fixed_focal_len

            
        else:
            pred_path = os.path.join(pred_folder, os.path.basename(names[0]).replace(".png", ".npy"))
            if verbal:
                print(iter, names[0], pred_path)
            try:
                pred_np = np.load(pred_path)			
                pred_np = cv2.resize(pred_np, (target_res[1], target_res[0]), interpolation=cv2.INTER_NEAREST)
                assert(fixed_focal_len is not None)
                pred_focal = fixed_focal_len

            except:
                try:
                    pred_np_f = np.load(pred_path, allow_pickle=True).item()
                    pred_np = pred_np_f["pred_depth"]
                    if fixed_focal_len is None:
                        pred_focal = pred_np_f["pred_focal"]
                    else:
                        pred_focal = fixed_focal_len
                except Exception as e:
                    run_log = "Error Loading %s: %s" % (os.path.basename(pred_path), str(e))
                    print(run_log)
                    return relative_error, metric_error_gt_no_order, run_log

        if b_vis_depth:
            vis_depth(pred_np, inputs.numpy(), step, target_res)

        #################### eval relative depth		
        if target_np is not None:
            _pred_np = np.expand_dims(pred_np, 0)
            _pred_np = np.expand_dims(_pred_np, 0)
            _pred_np = torch.from_numpy(_pred_np)
            report = evalute_correct_rate_one_img(_pred_np, target, [0.0], target_res)

            not_eq_correct_count += report[0.0]['not_eq_correct_count'].float()
            not_eq_count += report['not_eq_count'].float()


        if not b_eval_rel_depth_only:
            if gt_np is not None:
                orig_size_pred_np = cv2.resize(pred_np, (target_res[1], target_res[0]))
                
                ################################
                if fixed_focal_len is None:
                    scaling = (target_res[0] / pred_np.shape[0] + target_res[1] / pred_np.shape[1]) / 2.0
                    focal_pred *= scaling
                ################################

                try:
                    ##################################################################################
                    # metric 3: normalize the gt depth, no depth order considered
                    metric_error_gt_no_order = metric_gt_no_order(gt_np, orig_size_pred_np, 
                                                focal_pred = pred_focal, 
                                                focal_gt = focal_gt, 
                                                record = metric_error_gt_no_order, 
                                                )
                    ################################################################################## 


                except Exception as e:
                    print(str(e))
                    return relative_error, metric_error_gt_no_order, str(e)


        # this None assignment is necessary to keep the gpu memory clean
        target = None
        target_res = None

    
    print ("%d samples are evaluated.\n" % (iter))    
    if not b_eval_rel_depth_only:
        metric_error_gt_no_order["LSIV"] = metric_error_gt_no_order['loss_sum'] / metric_error_gt_no_order['n_pixel']
        print ("tentative focal length:", fixed_focal_len, "metric_error_gt_no_order LSIV Metric: %g" % (metric_error_gt_no_order["LSIV"]))
            
    correct_rate = float(not_eq_correct_count) / (float(not_eq_count) + 1e-7)
    relative_error['WKDR'] = (1.0 - correct_rate) * 100
    print("WKDR:", relative_error['WKDR'])
    return relative_error, metric_error_gt_no_order, "success"






