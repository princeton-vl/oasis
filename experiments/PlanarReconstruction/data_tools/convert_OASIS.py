"""
Convert OASIS dataset to PlanarReconstruction format.
"""
import cv2 as cv
import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle


def read_depth(path, h, w):
    f = open(path, 'rb')
    depth_dict = pickle.load(f)
    f.close()
    depth = - np.ones((h,w))
    min_y = depth_dict['min_y']
    max_y = depth_dict['max_y']

    min_x = depth_dict['min_x']
    max_x = depth_dict['max_x']
    
    roi_depth = depth_dict['depth']
    roi_depth[np.isnan(roi_depth)] = -1.0
    depth[min_y:max_y+1, min_x:max_x+1] = roi_depth
    depth = depth.astype(np.float32)
    return depth

def vis_mask(mask):
    out = np.copy(mask)
    out = out.astype(np.float32)
    bg_mask = out == 20
    # out[~bg_mask] = out[~bg_mask] - np.min(out[~bg_mask])
    out[~bg_mask] = out[~bg_mask] / np.max(out[~bg_mask])
    out[~bg_mask] *= 255.0

    return out

def read_normal(path, h, w):
    f = open(path, 'rb')
    normal_dict = pickle.load(f)
    f.close()
    normal = np.zeros((h,w,3))
    min_y = normal_dict['min_y']
    max_y = normal_dict['max_y']

    min_x = normal_dict['min_x']
    max_x = normal_dict['max_x']
    
    roi_normal = normal_dict['normal']
    #roi_depth[np.isnan(roi_depth)] = -1.0
    normal[min_y:max_y+1, min_x:max_x+1] = roi_normal
    normal = normal.astype(np.float32)
    return normal

def make_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok = True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='path to OASIS',
                        required=True)
    parser.add_argument('--train_csv', type=str,
                        help='path to train set csv',
                        required=True)
    parser.add_argument('--output_dir', type=str,
                        help='where to store extracted frames',
                        required=True)
    parser.add_argument('--data_type', type=str,
                        help='where to store extracted frames',
                        required=True)

    args = parser.parse_args()

    # read dataset csv
    dataset_csv = None
    if args.data_type == 'train':
        dataset_csv = args.train_csv
    else:
        raise ValueError
    f = open(dataset_csv, 'r')
    lines = f.readlines()
    f.close()
    lines.pop(0) # header

    # all dirs
    img_dir = os.path.join(args.dataset_path, 'image')
    depth_dir = os.path.join(args.dataset_path, 'depth')
    semantic_dir = os.path.join(args.dataset_path, 'segmentation/surface_semantic')
    instance_dir = os.path.join(args.dataset_path, 'segmentation/planar_instance')
    normal_dir = os.path.join(args.dataset_path, 'normal')
    npz_dir = os.path.join(args.output_dir, args.data_type)

    make_dir_if_not_exist(args.output_dir)
    make_dir_if_not_exist(npz_dir)

    # output txt file
    output_f = open(os.path.join(args.output_dir, args.data_type + '.txt'), 'w')

    for line in tqdm(lines):
        # start preparing data
        splits = line.split(',')
        anno = {}

        # anno['image']
        # shape: (192x256x3)
        # channels: BGR
        img_path = splits[0]        
        img_name = img_path.split('/')[-1]
        # img_name = '110603_DT.png'
        img_id = img_name.split('.')[0]        

        out_npz_path = os.path.join(npz_dir, img_id + '.npz')
        if os.path.exists(out_npz_path):
            tqdm.write("%s exists.." % (out_npz_path))

        img_path = os.path.join(img_dir, img_name)
        img = cv.imread(img_path)
        h = img.shape[0]
        w = img.shape[1]
        img = cv.resize(img, (256, 192))
        anno['image'] = img

        # anno['depth']
        # shape: (192, 256, 1)
        # Note: depth map
        try:
            depth = read_depth(os.path.join(depth_dir, img_id + '.pkl'), h, w)
        except Exception as e:
            tqdm.write(str(e))
            continue
        depth = cv.resize(depth, (256, 192), interpolation=cv.INTER_NEAREST)
        depth_mask = depth > 0
        depth = depth[:, :, np.newaxis]
        anno['depth'] = depth

        # anno['segmentation']
        # shape: (192, 256, 1)
        # Note: use 0-19 for each plane, use 20 for background
        #semantic_seg = cv.imread(os.path.join(semantic_dir, img_id + '.png'))
        instance_seg = cv.imread(os.path.join(instance_dir, img_id + '.png'))
        if instance_seg is None:
            tqdm.write("instance_seg is None: " + img_name)
            continue
        instance_seg = cv.resize(instance_seg, (256, 192), interpolation=cv.INTER_NEAREST) # (192, 256, 3)
        instance_seg[instance_seg > 20] = 0
        num_planes = instance_seg.max()
        unique_seg_ids = np.unique(instance_seg)

        #for i in range(1, num_planes + 1):
        #    if (instance_seg == i).sum() == 0:
        #        instance_seg[instance_seg > i] = instance_seg[instance_seg > i] - 1
        #        num_planes -= 1
        if num_planes == 0:
            tqdm.write("num_planes == 0: " + img_name)
            continue

        ################################################ reorder ids that are not continuous
        if (len(unique_seg_ids) - 1) != num_planes:
            tqdm.write("\trearranging order for %s ..."  % img_id)
            num_planes = len(unique_seg_ids) - 1    # -1 to exluce the background 0
            reorder_instance_seg = np.copy(instance_seg)
            reorder_instance_seg.fill(0)
            new_id = 0
            for seg_id in unique_seg_ids:
                if seg_id == 0:
                    tqdm.write("\t\torig 0 --> 20" )
                    reorder_instance_seg[instance_seg == 0] = 20
                    continue
                else:
                    tqdm.write("\t\torig %d --> %d" % (seg_id, new_id))
                    reorder_instance_seg[instance_seg == seg_id] = new_id
                    new_id += 1
            reorder_instance_seg = reorder_instance_seg[:, :, :1]            
            instance_seg = np.copy(reorder_instance_seg)
        else:
            instance_seg[instance_seg == 0] = 21
            instance_seg = instance_seg - 1
            instance_seg = instance_seg[:, :, :1]

        ######################## Find cases where the planar region and valid depth region does not overlap
        valid_region = instance_seg != 20
        valid_region = np.logical_and(depth_mask[:,:,np.newaxis], valid_region)
        if np.sum(valid_region) <= 10:
            tqdm.write("error: valid depth and planar region does not overlap " + img_name)
            continue


        #check
        flag_skip = False
        for i in range(20):
            if i < num_planes:
                if (instance_seg == i).sum() == 0:
                    tqdm.write("error " + img_name)
                    flag_skip = True
                    break
            else:
                if (instance_seg == i).sum() != 0:
                    tqdm.write("error2 " + img_name)
                    flag_skip = True
                    break
        if flag_skip:
            tqdm.write("flag_skip: " + img_name)
            continue
        
        anno['segmentation'] = instance_seg

        # anno['num_planes']
        # example: array([2], dtype=int32)
        # Note: number of planes
        anno['num_planes'] = np.array([num_planes])

        # anno['plane']
        # shape: (20, 3)
        # Note: define the surface normal of each plane
        normals = np.zeros((20, 3))
        #normal_map = cv.imread(os.path.join(normal_dir, img_id + '.png'))
        try:
            normal_map = read_normal(os.path.join(normal_dir, img_id + '.pkl'), h, w)
        except Exception as e:             
            tqdm.write(str(e))
            continue
        normal_map = cv.resize(normal_map, (256, 192), interpolation=cv.INTER_NEAREST)
        for i in range(num_planes):
            m = (instance_seg == i)[:, :, 0]
            if m.sum() == 0:
                tqdm.write("error 3 " + img_name)
                continue
            #normal = normal_map[m].mean()
            normal = normal_map[m].mean(axis=0)
            if np.isnan(normal).sum() != 0:
                tqdm.write("error 4 " + img_name)
                continue
            if np.linalg.norm(normal) < 0.01:
                tqdm.write("error 5 " + img_name)
                flag_skip = True
                break
            normals[i, :] = normal
            #normals[i, 2] = 1.0
        anno['plane'] = normals
        if flag_skip:
            continue

        # focal length
        focal_length = float(splits[1])
        anno['focal_length'] = focal_length

        # write
        np.savez(out_npz_path, **anno)
        output_f.write(img_id + '.npz\n')

    output_f.close()


if __name__=="__main__":
    main()
