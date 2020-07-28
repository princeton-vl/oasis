import os
from IPython import embed
import scipy.io as sio
import numpy as np
from scipy.misc import imread, imsave
import copy

test_folder = "./tests"
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

def save_in_eval_format(dst_dir, id, img):
    sio.savemat(os.path.join(dst_dir, '{}.mat'.format(id)), {'groundTruth': np.array([{'Boundaries': img}])})

def save_in_result_format(dst_dir, id, img):
    sio.savemat(os.path.join(dst_dir, '{}.mat'.format(id)), {'result': img})

# Test 1 (Basic Test)
def test1():
    # Prepare folders
    test1_folder = test_folder + "/test1"
    if not os.path.exists(test1_folder):
        os.mkdir(test1_folder)
    test1_boundary_folder = test_folder + "/test1/boundary"
    if not os.path.exists(test1_boundary_folder):
        os.mkdir(test1_boundary_folder)
    test1_boundary_folder = test_folder + "/test1/boundary/mat"
    if not os.path.exists(test1_boundary_folder):
        os.mkdir(test1_boundary_folder)
    test1_class_folder = test_folder + "/test1/class"
    if not os.path.exists(test1_class_folder):
        os.mkdir(test1_class_folder)
    test1_class_folder = test_folder + "/test1/class/mat"
    if not os.path.exists(test1_class_folder):
        os.mkdir(test1_class_folder)
    test1_boundary_gt_folder = test_folder + "/test1/boundary_eval_format"
    if not os.path.exists(test1_boundary_gt_folder):
        os.mkdir(test1_boundary_gt_folder)
    test1_occ_folder = test_folder + "/test1/occ_eval_format"
    if not os.path.exists(test1_occ_folder):
        os.mkdir(test1_occ_folder)
    test1_fold_folder = test_folder + "/test1/fold_eval_format"
    if not os.path.exists(test1_fold_folder):
        os.mkdir(test1_fold_folder)
    test1_mask_folder = test_folder + "/test1/mask_eval_format"
    if not os.path.exists(test1_mask_folder):
        os.mkdir(test1_mask_folder)
    # Generate Images
    boundary_image = np.zeros((5,5)) + np.eye(5)
    boundary_image[1,1] = 0
    class_image = np.zeros((5,5)) + np.eye(5)
    boundary_gt_image = np.zeros((5,5)) + np.eye(5)
    occ_gt_image = np.zeros((5,5)) + np.eye(5)
    fold_gt_image = np.zeros((5,5))
    mask_image = np.ones((5,5))
    # Save in folder
    save_in_result_format(test1_boundary_folder, 1,boundary_image)
    save_in_result_format(test1_class_folder, 1, class_image)
    save_in_eval_format(test1_boundary_gt_folder, 1, boundary_gt_image)
    save_in_eval_format(test1_occ_folder, 1, occ_gt_image)
    save_in_eval_format(test1_fold_folder, 1, fold_gt_image)
    save_in_result_format(test1_mask_folder, 1, mask_image)
    test1_png_folder = test_folder + "/test1/png/"
    if not os.path.exists(test1_png_folder):
        os.mkdir(test1_png_folder)
    imsave(test1_png_folder + "boundary.png", boundary_image)
    imsave(test1_png_folder + "class.png", class_image)
    imsave(test1_png_folder + "boundary_gt.png", boundary_gt_image)
    imsave(test1_png_folder + "occ_gt.png", occ_gt_image)
    imsave(test1_png_folder + "fold_gt.png", fold_gt_image)
    imsave(test1_png_folder + "mask.png", mask_image)


# Test 2 (ROI test)
def test2():
    # Prepare folders
    test2_folder = test_folder + "/test2"
    if not os.path.exists(test2_folder):
        os.mkdir(test2_folder)
    test2_boundary_folder = test_folder + "/test2/boundary"
    if not os.path.exists(test2_boundary_folder):
        os.mkdir(test2_boundary_folder)
    test2_boundary_folder = test_folder + "/test2/boundary/mat"
    if not os.path.exists(test2_boundary_folder):
        os.mkdir(test2_boundary_folder)
    test2_class_folder = test_folder + "/test2/class"
    if not os.path.exists(test2_class_folder):
        os.mkdir(test2_class_folder)
    test2_class_folder = test_folder + "/test2/class/mat"
    if not os.path.exists(test2_class_folder):
        os.mkdir(test2_class_folder)
    test2_boundary_gt_folder = test_folder + "/test2/boundary_eval_format"
    if not os.path.exists(test2_boundary_gt_folder):
        os.mkdir(test2_boundary_gt_folder)
    test2_occ_folder = test_folder + "/test2/occ_eval_format"
    if not os.path.exists(test2_occ_folder):
        os.mkdir(test2_occ_folder)
    test2_fold_folder = test_folder + "/test2/fold_eval_format"
    if not os.path.exists(test2_fold_folder):
        os.mkdir(test2_fold_folder)
    test2_mask_folder = test_folder + "/test2/mask_eval_format"
    if not os.path.exists(test2_mask_folder):
        os.mkdir(test2_mask_folder)
    # Generate Images
    boundary_image = np.zeros((5,5)) + np.eye(5)
    boundary_image[1,1] = 0
    class_image = np.zeros((5,5)) + np.eye(5)
    boundary_gt_image = np.zeros((5,5)) + np.eye(5)
    occ_gt_image = np.zeros((5,5)) + np.eye(5)
    fold_gt_image = np.zeros((5,5))
    mask_image = np.ones((5,5))
    mask_image[1,1] = 0
    # Save in folder
    save_in_result_format(test2_boundary_folder, 1,boundary_image)
    save_in_result_format(test2_class_folder, 1, class_image)
    save_in_eval_format(test2_boundary_gt_folder, 1, boundary_gt_image)
    save_in_eval_format(test2_occ_folder, 1, occ_gt_image)
    save_in_eval_format(test2_fold_folder, 1, fold_gt_image)
    save_in_result_format(test2_mask_folder, 1, mask_image)
    test2_png_folder = test_folder + "/test2/png/"
    if not os.path.exists(test2_png_folder):
        os.mkdir(test2_png_folder)
    imsave(test2_png_folder + "boundary.png", boundary_image)
    imsave(test2_png_folder + "class.png", class_image)
    imsave(test2_png_folder + "boundary_gt.png", boundary_gt_image)
    imsave(test2_png_folder + "occ_gt.png", occ_gt_image)
    imsave(test2_png_folder + "fold_gt.png", fold_gt_image)
    imsave(test2_png_folder + "mask.png", mask_image)


# Test 3 (Boudnary ODS vs OIS)
def test3():
    # Prepare folders
    test3_folder = test_folder + "/test3"
    if not os.path.exists(test3_folder):
        os.mkdir(test3_folder)
    test3_boundary_folder = test_folder + "/test3/boundary"
    if not os.path.exists(test3_boundary_folder):
        os.mkdir(test3_boundary_folder)
    test3_boundary_folder = test_folder + "/test3/boundary/mat"
    if not os.path.exists(test3_boundary_folder):
        os.mkdir(test3_boundary_folder)
    test3_class_folder = test_folder + "/test3/class"
    if not os.path.exists(test3_class_folder):
        os.mkdir(test3_class_folder)
    test3_class_folder = test_folder + "/test3/class/mat"
    if not os.path.exists(test3_class_folder):
        os.mkdir(test3_class_folder)
    test3_boundary_gt_folder = test_folder + "/test3/boundary_eval_format"
    if not os.path.exists(test3_boundary_gt_folder):
        os.mkdir(test3_boundary_gt_folder)
    test3_occ_folder = test_folder + "/test3/occ_eval_format"
    if not os.path.exists(test3_occ_folder):
        os.mkdir(test3_occ_folder)
    test3_fold_folder = test_folder + "/test3/fold_eval_format"
    if not os.path.exists(test3_fold_folder):
        os.mkdir(test3_fold_folder)
    test3_mask_folder = test_folder + "/test3/mask_eval_format"
    if not os.path.exists(test3_mask_folder):
        os.mkdir(test3_mask_folder)
    # Generate Images
    # Image 1: optimal threshhold of 0.5 <=  5 TP 0 FP 0 FN if 0.5 > 0 TP 0 FP 5 FN
    # Image 2: optimal threshhold of 0.6 >  0 TP 0 FP 4 FN if 0.6 <= 0 TP 5 FP 5 FN
    boundary_image_1 = np.zeros((5,5)) + np.eye(5) * 0.5
    class_image_1 = np.zeros((5,5)) + np.eye(5)
    boundary_gt_image_1 = np.zeros((5,5)) + np.eye(5)
    occ_gt_image_1 = np.zeros((5,5)) + np.eye(5)
    fold_gt_image_1 = np.zeros((5,5))
    mask_image_1 = np.ones((5,5))

    boundary_image_2 = np.zeros((5,5))
    boundary_image_2[0,1] = 0.6
    boundary_image_2[1,2] = 0.6
    boundary_image_2[0,2] = 0.6
    boundary_image_2[0,3] = 0.6
    boundary_image_2[0,4] = 0.6
    class_image_2 = np.zeros((5,5)) + np.eye(5)
    boundary_gt_image_2 = np.zeros((5,5)) + np.eye(5)
    occ_gt_image_2 = np.zeros((5,5)) + np.eye(5)
    fold_gt_image_2 = np.zeros((5,5))
    mask_image_2 = np.ones((5,5))

    # Save in folder
    save_in_result_format(test3_boundary_folder, 1,boundary_image_1)
    save_in_result_format(test3_class_folder, 1, class_image_1)
    save_in_eval_format(test3_boundary_gt_folder, 1, boundary_gt_image_1)
    save_in_eval_format(test3_occ_folder, 1, occ_gt_image_1)
    save_in_eval_format(test3_fold_folder, 1, fold_gt_image_1)
    save_in_result_format(test3_mask_folder, 1, mask_image_1)
    save_in_result_format(test3_boundary_folder, 2,boundary_image_2)
    save_in_result_format(test3_class_folder, 2, class_image_2)
    save_in_eval_format(test3_boundary_gt_folder, 2, boundary_gt_image_2)
    save_in_eval_format(test3_occ_folder, 2, occ_gt_image_2)
    save_in_eval_format(test3_fold_folder, 2, fold_gt_image_2)
    save_in_result_format(test3_mask_folder, 2, mask_image_2)

    test3_png_folder = test_folder + "/test3/png/"
    if not os.path.exists(test3_png_folder):
        os.mkdir(test3_png_folder)
    imsave(test3_png_folder + "boundary_1.png", boundary_image_1)
    imsave(test3_png_folder + "class_1.png", class_image_1)
    imsave(test3_png_folder + "boundary_gt_1.png", boundary_gt_image_1)
    imsave(test3_png_folder + "occ_gt_1.png", occ_gt_image_1)
    imsave(test3_png_folder + "fold_gt_1.png", fold_gt_image_1)
    imsave(test3_png_folder + "mask_1.png", mask_image_1)
    imsave(test3_png_folder + "boundary_2.png", boundary_image_2)
    imsave(test3_png_folder + "class_2.png", class_image_2)
    imsave(test3_png_folder + "boundary_gt_2.png", boundary_gt_image_2)
    imsave(test3_png_folder + "occ_gt_2.png", occ_gt_image_2)
    imsave(test3_png_folder + "fold_gt_2.png", fold_gt_image_2)
    imsave(test3_png_folder + "mask_2.png", mask_image_2)


# Test 4 (Class test)
def test4():
    # Prepare folders
    test4_folder = test_folder + "/test4"
    if not os.path.exists(test4_folder):
        os.mkdir(test4_folder)
    test4_boundary_folder = test_folder + "/test4/boundary"
    if not os.path.exists(test4_boundary_folder):
        os.mkdir(test4_boundary_folder)
    test4_boundary_folder = test_folder + "/test4/boundary/mat"
    if not os.path.exists(test4_boundary_folder):
        os.mkdir(test4_boundary_folder)
    test4_class_folder = test_folder + "/test4/class"
    if not os.path.exists(test4_class_folder):
        os.mkdir(test4_class_folder)
    test4_class_folder = test_folder + "/test4/class/mat"
    if not os.path.exists(test4_class_folder):
        os.mkdir(test4_class_folder)
    test4_boundary_gt_folder = test_folder + "/test4/boundary_eval_format"
    if not os.path.exists(test4_boundary_gt_folder):
        os.mkdir(test4_boundary_gt_folder)
    test4_occ_folder = test_folder + "/test4/occ_eval_format"
    if not os.path.exists(test4_occ_folder):
        os.mkdir(test4_occ_folder)
    test4_fold_folder = test_folder + "/test4/fold_eval_format"
    if not os.path.exists(test4_fold_folder):
        os.mkdir(test4_fold_folder)
    test4_mask_folder = test_folder + "/test4/mask_eval_format"
    if not os.path.exists(test4_mask_folder):
        os.mkdir(test4_mask_folder)
    # Generate Images
    boundary_image = np.zeros((5,5)) + np.eye(5)
    class_image = np.zeros((5,5))
    class_image[0,0] = 1
    class_image[1,1] = 1
    boundary_gt_image = np.zeros((5,5)) + np.eye(5)
    occ_gt_image = np.zeros((5,5)) + np.eye(5)
    fold_gt_image = np.zeros((5,5))
    mask_image = np.ones((5,5))
    # Save in folder
    save_in_result_format(test4_boundary_folder, 1,boundary_image)
    save_in_result_format(test4_class_folder, 1, class_image)
    save_in_eval_format(test4_boundary_gt_folder, 1, boundary_gt_image)
    save_in_eval_format(test4_occ_folder, 1, occ_gt_image)
    save_in_eval_format(test4_fold_folder, 1, fold_gt_image)
    save_in_result_format(test4_mask_folder, 1, mask_image)
    test4_png_folder = test_folder + "/test4/png/"
    if not os.path.exists(test4_png_folder):
        os.mkdir(test4_png_folder)
    imsave(test4_png_folder + "boundary.png", boundary_image)
    imsave(test4_png_folder + "class.png", class_image)
    imsave(test4_png_folder + "boundary_gt.png", boundary_gt_image)
    imsave(test4_png_folder + "occ_gt.png", occ_gt_image)
    imsave(test4_png_folder + "fold_gt.png", fold_gt_image)
    imsave(test4_png_folder + "mask.png", mask_image)

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
