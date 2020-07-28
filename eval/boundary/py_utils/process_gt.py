import os
# from IPython import embed
import scipy.io as sio
import cv2
import numpy as np
from imageio import imread, imsave
import argparse
import pickle
import glob
# from scipy.misc import imread
# from scipy.misc import imread, imsave


# Convert occlusion and fold ground truth into a matlab structure which is compatibe with the Pitor's evalaution code.
def convert_gt_to_eval_format(file_list, src_dir, dst_dir="./data/3SIW_fold/fold_eval_format"):
    print("Convert gt to eval format...")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        
    for i, img in enumerate(file_list):
        if ".png" in img:
            try:
                img = os.path.basename(img)
                output_name = img.replace(".png", "")

                src_path = os.path.join(src_dir, img)
                gt = imread(src_path)
                gt = gt[:,:,0]
                gt[gt > 240]=1
                sio.savemat(os.path.join(dst_dir, '{}.mat'.format(output_name)), {'groundTruth': np.array([{'Boundaries': gt}])})
                if i % 100 == 0:
                    print("{} th image processed.".format(i))
            except Exception as e:
                print("WARNING: {} failed.".format(img))
                print('Message: ' + str(e))
                pass

# Convert mask into .mat format.
def convert_mask_to_eval_format(file_list, src_dir, dst_dir="./data/3SIW_fold/mask_eval_format"):
    print("Convert mask to eval format...")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for i, img in enumerate(file_list):
        if ".png" in img:
            try:
                img = os.path.basename(img)
                output_name = img.replace(".png", "")

                src_path = os.path.join(src_dir, img)
                gt = imread(src_path)
                gt = gt[:,:,0]
                gt[gt > 240]=1
                sio.savemat(os.path.join(dst_dir, '{}.mat'.format(output_name)), {'ROI': gt})
                if i % 100 == 0:
                    print("{} th image processed.".format(i))
            except Exception as e:
                print("WARNING: {} failed.".format(img))
                print('Message: ' + str(e))
                pass

# # Combine smooth and sharp occlusion.
# def combine_occ(src_dir = "./data/3SIW_occ", dst_dir="./data/3SIW_occ/occ"):
#     sharp_occ_folder = os.path.join(src_dir, "sharpocc")
#     smooth_occ_folder = os.path.join(src_dir, "smoothocc")
#     if not os.path.exists(dst_dir):
#         os.mkdir(dst_dir)
#     for i, fn in enumerate(os.listdir(sharp_occ_folder)):
#         if ".png" in fn:
#             try:
#                 sharp_img = imread(os.path.join(sharp_occ_folder, fn))
#                 smooth_img = imread(os.path.join(smooth_occ_folder, fn))
#                 occ_img = sharp_img + smooth_img
#                 occ_img = np.clip(occ_img, 0,255)
#                 imsave("{}/{}".format(dst_dir, fn), occ_img)
#                 if i % 100 == 0:
#                     print("{} th image processed.".format(i))
#             except:
#                 print("WARNING: {} failed.".format(fn))
#                 # 
#                 pass


# Combine smooth and sharp occlusion.
def create_boundary(occ_files, fold_files, src_dir, dst_dir="./data/3SIW_occ_fold/boundary"):
    print("Create boundary...")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for i, (occ, fold) in enumerate(zip(occ_files, fold_files)):
        if ".png" in occ:
            try:
                output_name = os.path.basename(occ).replace(".png", "")

                occ_img = imread(os.path.join(src_dir, "occ", os.path.basename(occ)))
                fold_img = imread(os.path.join(src_dir, "fold",  os.path.basename(fold)))

                cmb_img = occ_img + fold_img
                cmb_img = np.clip(cmb_img, 0,255)
                imsave("{}/{}.png".format(dst_dir, output_name), cmb_img)
                if i % 100 == 0:
                    print("{} th image processed.".format(i))
            except Exception as e:
                print("WARNING: {} failed.".format(occ))
                print('Message: ' + str(e))
                pass


def load_obj(name, verbal=False):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
        if verbal:
            print(" Done loading %s" % name)
        return obj



# python process_gt.py --base_dir ../../../nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold_trainval --pkl validation_splits.pkl
# python process_gt.py --base_dir ../../../nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold_test  --pkl test_splits.pkl
if __name__ == '__main__':

    # the base_dir should be the data/OASIS_occ_fold under nori_training_tmp/hed_xwjabc
    parser = argparse.ArgumentParser(description='OASIS occ fold data preprocessing.')
    parser.add_argument('--base_dir',  default="../../../nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold_trainval", type=str)    
    parser.add_argument('--pkl',  default="validation_splits.pkl", type=str)
    args = parser.parse_args()

    
    to_process_data = load_obj("%s/%s" % (args.base_dir, args.pkl))
    print("# val data: {}".format(len(to_process_data)))

    fold_files = [elem["fold"] for elem in to_process_data]    # + [elem["fold"] for elem in test_data]
    occ_files = [elem["occ"] for elem in to_process_data]      # + [elem["occ"] for elem in test_data]
    mask_files = [elem["mask"] for elem in to_process_data]    # + [elem["mask"] for elem in test_data]


    convert_gt_to_eval_format(fold_files, \
                              src_dir="%s/fold" % args.base_dir,\
                              dst_dir="%s/fold_eval_format" % args.base_dir)
    convert_gt_to_eval_format(occ_files, \
                              src_dir="%s/occ" % args.base_dir, \
                              dst_dir="%s/occ_eval_format" % args.base_dir)
    convert_mask_to_eval_format(mask_files, \
                              src_dir="%s/mask" % args.base_dir, \
                              dst_dir="%s/mask_eval_format" % args.base_dir)
    
    create_boundary(occ_files = occ_files, 
                    fold_files = fold_files,
                    src_dir = args.base_dir,
                    dst_dir = "%s/boundary" % args.base_dir)
    

    boundary_files = glob.glob("%s/boundary/*.png" % args.base_dir)
    convert_gt_to_eval_format(boundary_files, \
                              src_dir="%s/boundary" % args.base_dir, \
                              dst_dir="%s/boundary_eval_format" % args.base_dir)
    
    # combine_occ()
