# Depth Estimation

## Data Preparation

```bash
PATH_TO_OASIS_TEST_AND_TRAINVAL=XXX
ln -s $PATH_TO_OASIS_TEST_AND_TRAINVAL OASIS
```


## Test

To reproduce the results present in the paper, first download the pretrained models, then run: 

```bash
python inference.py -out_dir ./depth -t OASIS/OASIS_test/OASIS_test.csv -model exp/2ReDWebNetReluMin_lr1e-05_LocalBackprojLoss2/models/best_sivmodel_iter_36000.bin

tar -czf depth.tar.gz depth
```

Then submit depth.tar to the [OASIS website](https://pvl.cs.princeton.edu/OASIS) to evaluate on the test set.

Below shows an example of running on the validation set.

```bash
# The LSIV_RMSE metric. 
python test2.py -iter 500 -t OASIS/OASIS_trainval/OASIS_val.csv -model exp/2ReDWebNetReluMin_lr1e-05_LocalBackprojLoss2/models/best_sivmodel_iter_36000.bin
# The WKDR metric.
python test2.py -iter 500 --DIW_rel_depth -t OASIS/OASIS_trainval/OASIS_val.csv -model exp/2ReDWebNetReluMin_lr1e-05_LocalBackprojLoss2/models/best_sivmodel_iter_36000.bin
```


## Train

To train the Hourglass network on OASIS depth from scratch, run:

```bash
python train2.py --num_iters 500000 -t OASIS/OASIS_trainval/OASIS_train.csv -v OASIS/OASIS_trainval/OASIS_val.csv -nlw 2 -lr 0.0001 -mn NIPS --loss_name LocalBackprojLoss2 -bs 12 --exp_name Hourglass
```

To train the ResNetD network on OASIS depth from scratch, run:

```bash
python train2.py --num_iters 500000 -t OASIS/OASIS_trainval/OASIS_train.csv -v OASIS/OASIS_trainval/OASIS_val.csv -nlw 2 -lr 1e-05 -mn ReDWebNetReluMin_raw --loss_name LocalBackprojLoss2 -bs 12 --exp_name ReDWebNetReluMin_raw
```

To finetune the ResNetD network (pretrained on ILSVRC) on OASIS depth, run:

```bash
python train2.py --num_iters 500000 -t OASIS/OASIS_trainval/OASIS_train.csv -v OASIS/OASIS_trainval/OASIS_val.csv -nlw 2 -lr 1e-05 -mn ReDWebNetReluMin --loss_name LocalBackprojLoss2 -bs 12  --exp_name ReDWebNetReluMin
```

Note that the model checkpoints will be saved in the `exp` folder.