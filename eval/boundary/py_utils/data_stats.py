import numpy as np
import os
from IPython import embed
from scipy.misc import imread

occ_path = "/n/fs/pvl/nkojima/p-surface/p-surfaces/nori_training_tmp/hed_xwjabc/data/3SIW_occ/occ"
fold_path = "/n/fs/pvl/nkojima/p-surface/p-surfaces/nori_training_tmp/hed_xwjabc/data/3SIW_fold/fold"


train_total_ct, val_total_ct, test_total_ct = 0,0,0
occ_train_zero_ct, occ_val_zero_ct, occ_test_zero_ct = 0,0,0
fold_train_zero_ct, fold_val_zero_ct, fold_test_zero_ct = 0,0,0

for i, fn in enumerate(os.listdir(occ_path)):
    if '.png' in fn:
        if i % 100 == 0:
            print(i)
        id = int(fn.replace('.png', ''))
        occ_fn_path = "{}/{}".format(occ_path, fn)
        fold_fn_path = "{}/{}".format(fold_path, fn)
        try:
            occ_gt = imread(occ_fn_path)
            fold_gt = imread(fold_fn_path)
        except:
            print("either occlusion or fold failed.")
        if id < 12159:
            train_total_ct += 1
            if np.sum(occ_gt) == 0:
                occ_train_zero_ct += 1
            if np.sum(fold_gt) == 0:
                fold_train_zero_ct += 1
        elif 14975 <= id:
            test_total_ct += 1
            if np.sum(occ_gt) == 0:
                occ_test_zero_ct += 1
            if np.sum(fold_gt) == 0:
                fold_test_zero_ct += 1
        else:
            val_total_ct += 1
            if np.sum(occ_gt) == 0:
                occ_val_zero_ct += 1
            if np.sum(fold_gt) == 0:
                fold_val_zero_ct += 1
print(train_total_ct, val_total_ct, test_total_ct, occ_train_zero_ct, occ_val_zero_ct, occ_test_zero_ct, fold_train_zero_ct, fold_val_zero_ct, fold_test_zero_ct)
print("Occ stats, Train: {}%, Val {}%, Test {}%".format(occ_train_zero_ct/train_total_ct*100,\
occ_val_zero_ct/val_total_ct*100, occ_test_zero_ct/test_total_ct*100))

print("Fold stats, Train: {}%, Val {}%, Test {}%".format(fold_train_zero_ct/train_total_ct*100,\
fold_val_zero_ct/val_total_ct*100, fold_test_zero_ct/test_total_ct*100))
