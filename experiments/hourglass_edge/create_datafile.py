import csv
import numpy as np
import os
import pickle
import argparse

def save_obj(obj, name, verbal=False):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbal:
            print(" Done saving %s" % name)

def parse_csv(filename):
    out_ids = []
    with open(filename, "r", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
                if i == 0:  # skip the first line which is the header
                    continue
                img_id = row[0]
                id = row[0].split('/')[-1].replace(".png", "")
                img_id = "image/%s.png" % id
                mask_id = img_id.replace("image", "mask")
                fold_id = img_id.replace("image", "fold")
                occ_id  = img_id.replace("image", "occ")
                out_ids.append({"img": img_id, "mask": mask_id, "occ":occ_id, "fold":fold_id})
    return out_ids



# python create_datafile.py --train_csv /home/wfchen/OASIS_trainval/csv_after_clean/OASIS_train.csv \
#                           --val_csv /home/wfchen/OASIS_trainval/csv_after_clean/OASIS_val.csv\
#                           --test_csv /home/wfchen/OASIS_trainval/OASIS_test.csv 
if __name__ == "__main__":
    # the base_dir should be the data/OASIS_occ_fold under nori_training_tmp/hed_xwjabc
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_csv',  default="OASIS_training_files.csv", type=str)
    parser.add_argument('--test_csv',  default="OASIS_testing_files.csv", type=str)
    parser.add_argument('--val_csv',  default="OASIS_validation_files.csv", type=str)
    args = parser.parse_args()

    # Train
    train_ids = parse_csv(args.train_csv)
    print("Train: {}".format(len(train_ids)))
    save_obj(train_ids, "data/OASIS_occ_fold_trainval/train_splits.pkl")

    # Val
    val_ids = parse_csv(args.val_csv)
    print("Val: {}".format(len(val_ids)))
    save_obj(val_ids, "data/OASIS_occ_fold_trainval/validation_splits.pkl")

    # Test
    test_ids = parse_csv(args.test_csv)
    print("Test: {}".format(len(test_ids)))
    save_obj(test_ids, "data/OASIS_occ_fold_test/test_splits.pkl")

    # Occlusion + Fold (Multi-class)
    with open("data/OASIS_occ_fold_trainval/train.txt","w") as f:
        for elem in train_ids:
            f.write("%s, %s, %s, %s\n" % (elem["img"], elem["mask"], elem["occ"], elem["fold"]))

    with open("data/OASIS_occ_fold_trainval/val.txt","w") as f:
        for elem in val_ids:
            f.write("%s, %s, %s, %s\n" % (elem["img"], elem["mask"], elem["occ"], elem["fold"]))

    with open("data/OASIS_occ_fold_test/test.txt","w") as f:
        for elem in test_ids:
            f.write("%s, %s, %s, %s\n" % (elem["img"], elem["mask"], elem["occ"], elem["fold"]))