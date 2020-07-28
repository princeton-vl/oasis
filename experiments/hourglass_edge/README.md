# Fold & Occlusion Boundary Detection using the Hourglass Network


## Prepare the OASIS Dataset

```bash
mkdir OASIS
PATH_TO_OASIS_TRAINVAL=XXX
PATH_TO_OASIS_TEST=XXX
ln -s $PATH_TO_OASIS_TEST OASIS/OASIS_test
ln -s $PATH_TO_OASIS_TRAINVAL OASIS/OASIS_trainval


mkdir -p data/OASIS_occ_fold_trainval
ln -s $PATH_TO_OASIS_TRAINVAL/image       data/OASIS_occ_fold_trainval/image
ln -s $PATH_TO_OASIS_TRAINVAL/occlusion   data/OASIS_occ_fold_trainval/occ
ln -s $PATH_TO_OASIS_TRAINVAL/fold        data/OASIS_occ_fold_trainval/fold
ln -s $PATH_TO_OASIS_TRAINVAL/mask        data/OASIS_occ_fold_trainval/mask

mkdir -p data/OASIS_occ_fold_test
ln -s $PATH_TO_OASIS_TEST/image       data/OASIS_occ_fold_test/image
ln -s $PATH_TO_OASIS_TEST/occlusion   data/OASIS_occ_fold_test/occ
ln -s $PATH_TO_OASIS_TEST/fold        data/OASIS_occ_fold_test/fold
ln -s $PATH_TO_OASIS_TEST/mask        data/OASIS_occ_fold_test/mask

# Create train_splits.pkl, validation_splits.pkl, train.txt and val.txt under folder data/OASIS_occ_fold_trainval, and test_splits.pkl and test.txt under data/OASIS_occ_fold_test
python create_datafile.py --train_csv $PATH_TO_OASIS_TRAINVAL/OASIS_train.csv \
                          --val_csv   $PATH_TO_OASIS_TRAINVAL/OASIS_val.csv\
                          --test_csv  $PATH_TO_OASIS_TEST/OASIS_test.csv 

# Convert all the gt from image into .mat format. It only processes files specified in validation_splits.pkl and test_splits.pkl
cd ../../eval/boundary/py_utils/
python process_gt.py --base_dir ../../../experiments/hourglass_edge/data/OASIS_occ_fold_trainval --pkl validation_splits.pkl
```


## Test

To reproduce the results presented in the paper, first download the pretrained model, then run:

```bash
# This command puts all the inference results into exps/HG_lr1e-05_a5/epoch-14-test.
# Note that the --upload flag stores the data in the output resolution of the network, which would be resized to the same resolution as the input image on the evaluation server. 
python hourglass_multi_train.py --dataset_name OASIS_occ_fold_test --test --nstack 1 --checkpoint exps/HG_lr1e-05_a5/epoch-14-checkpoint.pt --exp_name HG_lr1e-05_a5 --upload

mv exps/HG_lr1e-05_a5/epoch-14-test occfold
tar -czf occfold.tar.gz occfold
```

Then submit `occfold.tar.gz` to the [OASIS website](https://pvl.cs.princeton.edu/OASIS) to evaluate on our test set.

Below shows an example of running on the validation set.

```bash
python hourglass_multi_train.py --dataset_name OASIS_occ_fold_trainval --val --nstack 1 --checkpoint exps/HG_lr1e-05_a5/epoch-14-checkpoint.pt --exp_name HG_lr1e-05_a5

cd ../eval/boundary
(echo "gt_data_folder = '../../experiments/hed_xwjabc/data/OASIS_occ_fold_trainval'"; echo "boundary_dir = '../../experiments/hourglass_edge/exps/HG_lr1e-05_a5/epoch-14-val/boundary'"; echo "class_dir = '../../experiments/hourglass_edge/exps/HG_lr1e-05_a5/epoch-14-val/class'"; cat eval_edge_multi.m)|matlab -nodisplay -nodesktop -nosplash
```

Note that the evaluation code is computation-intensive. The evaluation process takes ~10 hrs with 20 Intel Core i7-5930K CPU @ 3.50GHz.


## Train

To replicate the model used in the paper, run:

```bash
EXP_NAME=HG_lr1e-05_a5
python hourglass_multi_train.py --dataset_name OASIS_occ_fold_trainval --exp_name $EXP_NAME --max_epoch 30 --nstack 1 --lr 1e-05 --alpha 5
```

The trained models are stored in `exp/EXP_NAME` folder. Note that after each epoch, the code will automatically run inference on the validation set and store the results into folders called `epoch-X-val`. You then need to evaluate them using a command like this:

```bash
EXP_NAME=XXX
ITER=XXX
cd ../eval/boundary
(echo "gt_data_folder = '../../experiments/hed_xwjabc/data/OASIS_occ_fold_trainval'"; echo "boundary_dir = '../../experiments/hourglass_edge/exps/$EXP_NAME/epoch-$ITER-val/boundary'"; echo "class_dir = '../../experiments/hourglass_edge/exps/$EXP_NAME/epoch-$ITER-val/class'"; cat eval_edge_multi.m)|matlab -nodisplay -nodesktop -nosplash
```

There are several parameters you might want to play around with.
- nstack:  the number of stacks in a hourgalass model
- lr: the learning rate
- alpha: the factor which will be multiplied to a classification loss. (total_loss = boundary_loss + alpha * classification_loss)


## Acknowledgment

The code is based on the PyTorch reimplementation of [Stacked Hourglass Networks](https://github.com/princeton-vl/pytorch_stacked_hourglass).