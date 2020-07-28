from semantic_surface_segmentation.eval_semantic_segmentation import evaluate_semantic_segmentation
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os


def evaluate(dataset_csv, pred_folder, gt_folder):
    f = open(dataset_csv, 'r')
    lines = f.readlines()
    f.close()
    lines.pop(0) # header
    #lines = [lines[0]] # debug
    #val = range(12159, 14975)
    #test = range(14975, 17408)
    mean_ious = []
    pred_segs = []
    gt_segs = []
    cnt = 0
    #for img_id in tqdm(test):
    for line in tqdm(lines):
        splits = line.split(',')
        img_path = splits[0]
        img_name = img_path.split('/')[-1]
        img_id = img_name.split('.')[0]
        img_path = os.path.join(pred_folder, '{}.png'.format(img_id))
        #img_name = img_path.split('/')[-1]
        try: # in case there is no gt
            gt_mask = imageio.imread(os.path.join(gt_folder, img_name))
        except:
            #print("skip {}".format(img_id))
            continue

        cnt += 1

        # if gt only has 0, mean_iou = nan
        #if (gt_mask == 1).sum() == 0 and (gt_mask == 2).sum() == 0:
        #    continue

        # upsample prediction
        h, w = gt_mask.shape
        pred_mask = imageio.imread(img_path)
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
        pred_mask[pred_mask >= 1] = 1
        pred_mask[pred_mask < 1] = 2

        #pred_mask = np.ones_like(gt_mask) * 2

        # evaluate
        pred_segs.append(pred_mask)
        gt_segs.append(gt_mask)

    global_acc, mean_acc, planar_acc, curved_acc, mean_iou, planar_iou, curved_iou, fw_iou = evaluate_semantic_segmentation(pred_segs=pred_segs, gt_segs=gt_segs)
    print(mean_iou)
    print(planar_iou)
    print(curved_iou)
    print(cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_csv')
    parser.add_argument('pred_folder')
    parser.add_argument('gt_folder')
    args = parser.parse_args()
    evaluate(args.dataset_csv, args.pred_folder, args.gt_folder)


if __name__=="__main__":
    main()
