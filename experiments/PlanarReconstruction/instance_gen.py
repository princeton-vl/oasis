import imageio
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os


# Usage: python instance_gen.py --gt_root_folder ./data/OASIS_test --dataset_csv OASIS_test.csv --pred_folder ./ae_outputs/
# the output will be stored under the `mask` and `seg` folder of pred_folder.
parser = argparse.ArgumentParser(description='HED training.')
parser.add_argument('--dataset_csv',  default='OASIS_test.csv')
parser.add_argument('--gt_root_folder',  default='./data/OASIS_test')
parser.add_argument('--pred_folder',  default='./ae_outputs/')
args = parser.parse_args()


gt_folder = os.path.join(args.gt_root_folder, 'segmentation/planar_instance/')
pred_folder = args.pred_folder
roi_folder = os.path.join(args.gt_root_folder, 'mask')
dataset_csv = os.path.join(args.gt_root_folder, args.dataset_csv)


def generate():
    f = open(dataset_csv, 'r')
    lines = f.readlines()
    f.close()
    lines.pop(0) # header

    pred_segs = []
    gt_segs = []

    with open(os.path.join(pred_folder, 'test_list.txt'), 'w') as f:
        for line in tqdm(lines):
            splits = line.split(',')
            img_path = splits[0]
            img_name = img_path.split('/')[-1]
            #img_name = '7618.png'
            img_id = img_name.split('.')[0]
            img_path = os.path.join(pred_folder, 'seg', '{}.png'.format(img_id))
            img_name = img_path.split('/')[-1]
            try: # in case there is no gt
                gt_path = os.path.join(gt_folder, img_name)
                gt_mask = imageio.imread(gt_path)
            except:
                print("Warning: gt for %s is missing" % img_name )
                continue

            # upsample prediction
            #h, w = gt_mask.shape
            #pred_mask = imageio.imread(img_path)
            #pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            #pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
            #pred_mask[pred_mask >= 1] = 2
            #pred_mask[pred_mask < 1] = 1

            roi_path = os.path.join(roi_folder, img_name)

            #pred_segs.append(pred_mask)
            #gt_segs.append(gt_mask)
            f.write(os.path.abspath(gt_path) + ',' + os.path.abspath(roi_path) + '\n')

        #mean_ious = np.array(mean_ious)
        #print(mean_ious.mean())
        #global_acc, mean_acc, planar_acc, curved_acc, mean_iou, planar_iou, curved_iou, fw_iou = evaluate_semantic_segmentation(pred_segs=pred_segs, gt_segs=gt_segs)
        #print(mean_iou)


if __name__=="__main__":
    generate()
