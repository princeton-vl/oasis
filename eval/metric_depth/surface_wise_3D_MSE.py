import numpy as np
from numpy.linalg import inv
import random
import cv2
import os


import pickle


def create_obj_files(filename, XYZ, color = [1,0,0]):
    n_pts = len(XYZ)
    with open(filename, "w") as f:
        for i in range(n_pts):
            f.write("v %g %g %g %g %g %g\n" % (XYZ[i][0], XYZ[i][1], XYZ[i][2], color[0], color[1], color[2]))
    

def load_obj(name, verbal=False):
    with open(name, 'rb') as f:
        try:
            obj = pickle.load(f)
        except:
            obj = pickle.load(f, encoding='bytes')
        if verbal:
            print( " Done loading %s" % name)
        return obj


def save_obj(obj, name, verbal=False ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbal:
            print( " Done saving %s" % name)


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


def closed_form_solution(pred, gt):
    '''
        pred:   Nx3 XYZ
        gt:     Nx3 XYZ

    '''
    # print("closed form solutions")	    
    assert(pred.shape[1] == 3)
    assert(gt.shape[0] == pred.shape[0])

    N_pt = gt.shape[0] + 1e-8

    x1 = gt[:,0]
    y1 = gt[:,1]
    z1 = gt[:,2]

    x2 = pred[:,0]
    y2 = pred[:,1]
    z2 = pred[:,2]

    X1_2 = np.sum(np.square(x1))
    Y1_2 = np.sum(np.square(y1))
    Z1_2 = np.sum(np.square(z1))
    X1X2 = np.sum(np.multiply(x1, x2))
    Y1Y2 = np.sum(np.multiply(y1, y2))
    Z1Z2 = np.sum(np.multiply(z1, z2))

    Z1_sum = np.sum(z1)
    Z2_sum = np.sum(z2)

    # z2x1_2 = np.sum(np.multiply(z2, np.square(x1)))
    # z2y1_2 = np.sum(np.multiply(z2, np.square(y1)))
    # z1x1x2 = np.sum(np.multiply(z1, np.multiply(x1, x2)))
    # z1y1y2 = np.sum(np.multiply(z1, np.multiply(y1, y2)))

    # denominator = X1_2 + Y1_2 + 1e-8
    # scale = (X1X2 + Y1Y2) / denominator
    # delta = (z2x1_2 + z2y1_2 - z1x1x2 - z1y1y2) / denominator
    
    denominator = X1_2 + Y1_2 + Z1_2 - Z1_sum * Z1_sum / N_pt
    scale2 = (X1X2 + Y1Y2 + Z1Z2 - Z1_sum * Z2_sum / N_pt) / denominator
    delta2 = (Z2_sum - scale2 * Z1_sum) / N_pt


    # loss = np.sum(np.power(np.matmul(A, [[scale], [delta]]) - b,2))

    # temp = scale * gt     
    # temp[:,-1] += delta    
    # loss = np.sum(np.square( temp - pred ))
    # print(loss, scale, delta)

    temp = scale2 * gt     
    temp[:,-1] += delta2 
    loss2 = np.sum(np.square( temp - pred ))
        
    return loss2, {1:scale2, 'translation':delta2}

def back_project(depth, f, surface_ids, b_normalize = False):
    height, width  = depth.shape[:2]

    xs = np.linspace(0, width-1, width)
    ys = np.linspace(0, height-1, height)
    xv, yv = np.meshgrid(xs, ys)
    
    mask = surface_ids > 0

    xv -= 0.5 * width
    yv -= 0.5 * height

    if f > 100000000:
        print("Orthographic Projection!")
        X = xv[mask] 
        Y = yv[mask] 
        Z = depth[mask]

    else:
        X = xv[mask] / (f + 1e-8) * depth[mask]
        Y = yv[mask] / (f + 1e-8) * depth[mask]
        Z = depth[mask]

    XYZ = np.stack([X,Y,Z]).transpose()   # N by 3
    
    if b_normalize:
        sigma = np.std(X, ddof = 1)
        XYZ /= (sigma + 1e-8)
    
    new_surface_id = np.stack([surface_ids[mask], surface_ids[mask], surface_ids[mask]])
    new_surface_id = new_surface_id.transpose()

    return XYZ, new_surface_id

def obtain_surface_ids(gt_depth):
    valid_mask = (gt_depth > 0).astype(np.uint8)
    
    _, surface_ids = cv2.connectedComponents(image = valid_mask, connectivity = 4, ltype = cv2.CV_32S)
    surface_ids = surface_ids.astype(np.uint8)	# 0 is the background	
    surface_ids[gt_depth <= 0] = 0

    # clean
    for id in np.unique(surface_ids):
        if np.sum(surface_ids == id) < 10:
            print("Surface %d has less than 10 pixels" % id)
            surface_ids[surface_ids == id] = 0

    assert(len(surface_ids.shape) == 2)

    return surface_ids



def surface_wise_3D_MSE_no_order(pred, gt, focal_pred, focal_gt, surface_ids, mesh_id = None, b_keep_neg = False):
    '''
    Calculate the locally scale-invariant error for pred depth map.
        closed-form scaling and translating for each individual surface. No depth ordering considered.

    Input:
        pred : a numpy array of HxW, np.float32.
        gt   : a numpy array of HxW, np.float32. The invalid pixels must be marked as negative!!!!
        focal_pred: the predicted focal length, a float number
        focal_gt  : the gt focal length, a float number
        mesh_id: if is not none, then output mesh
    Output:
        Error: the minimal error that can be achieved under the depth constraint specified in depth_pairs
        The solved scaling fator the results in minimal erro
    '''
    assert(pred.shape[0] == gt.shape[0] and pred.shape[1] == gt.shape[1])

    gt = np.copy(gt)
    pred = np.copy(pred)

    n_pixel = float(np.sum(surface_ids > 0))



    ################################################################
    # back project and normalize the predicted depth
    XYZ_pred, _          = back_project(pred, focal_pred, surface_ids, b_normalize = True)
    XYZ_gt,surface_ids   = back_project(gt, focal_gt, surface_ids, b_normalize = False)

    # make the id continous
    unq_ids = [i for i in np.unique(surface_ids) if i != 0]
    loss_sum = 0.0
    scales_2_s_id = {}
    for s_id in unq_ids:
        mask = surface_ids == s_id
        cf_loss, solution = closed_form_solution(XYZ_pred[mask].reshape(-1, 3), XYZ_gt[mask].reshape(-1, 3))
        loss_sum += cf_loss
        scales_2_s_id[s_id] = solution

    
    if mesh_id is not None:
        list_XYZ_pred = []
        for i in range(XYZ_pred.shape[0]):
            list_XYZ_pred.append(XYZ_pred[i, :]) 
        create_obj_files('./%s_pred.obj' % mesh_id, list_XYZ_pred, color = [1,0,0])

        list_XYZ_gt = []
        for i in range(XYZ_gt.shape[0]):
            list_XYZ_gt.append(XYZ_gt[i, :]) 
        create_obj_files('./%s_gt_before.obj' % mesh_id, list_XYZ_gt, color = [1,0,1])

        list_XYZ_gt = []        
        for s_id in unq_ids:
            mask = (surface_ids == s_id)        
            solution = scales_2_s_id[s_id]    
            scale = solution[1]
            translation = solution['translation']
            temp = scale * XYZ_gt[mask]  
            temp = temp.reshape(-1, 3) 
            temp[:,-1] += translation
            
            for i in range(temp.shape[0]):
                list_XYZ_gt.append(temp[i, :]) 
        create_obj_files('./%s_gt_after_%g_%s.obj' % (mesh_id, loss_sum / n_pixel, 'iter'), list_XYZ_gt, color = [0,1,0])
    

    return {'loss_sum':loss_sum,  'n_pixel':n_pixel}

