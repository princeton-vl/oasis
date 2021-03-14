# Normal Estimation 

## Data Preparation

```bash
PATH_TO_OASIS_TEST_AND_TRAINVAL=XXX
ln -s $PATH_TO_OASIS_TEST_AND_TRAINVAL OASIS
```

## Test

To reproduce the results present in the paper, first download the pretrained models, then run: 

```bash
python inference.py --out_folder normals --test_csv OASIS/OASIS_test/OASIS_test.csv --model exp/nom_OASIS_0.0001/models/best_model_iter_249000.bin

tar -czf normals.tar.gz normals
```

Then submit normals.tar.gz to the [OASIS website](https://pvl.cs.princeton.edu/OASIS) to evaluate on the test set.

Below shows an example of running on the validation set.

```bash
EXP_NAME=nom_OASIS_0.0001
MODEL_ITER=189000
# Absoulute normal evaluation
python test.py -iter 500 -t OASIS/OASIS_trainval/OASIS_val.csv -model exp/$EXP_NAME/models/best_model_iter_$MODEL_ITER.bin -in_coord_sys OASIS -out_coord_sys OASIS
# Relative normal evaluation
python relative_normal_eval/eval_rel.py --test_file relative_normal_eval/rel_normal_pair_val.csv --model_file exp/$EXP_NAME/models/model_iter_$MODEL_ITER.bin -in_coord_sys OASIS --out_coord_sys OASIS --exp_name $EXP_NAME
```
The file `rel_normal_pair_val.csv` can be downloaded from this [link](https://drive.google.com/file/d/1LRfITcQ8va6m1S7emlCAGC11iEemxe9S/view?usp=sharing).

## Train

To replicate the Hourglass network on OASIS normals as in the paper, run:

```bash
EXP_NAME=nom_OASIS_0.0001
python train.py -iter 500000 -t OASIS/OASIS_trainval/OASIS_train.csv -v OASIS/OASIS_trainval/OASIS_val.csv -dn OASISNormalDataset -lr 0.0001 -bs 12 -mn NIPSSurface --loss_name 'CosineAngularLoss'  -in_coord_sys OASIS -out_coord_sys OASIS --exp_name $EXP_NAME
```

Note that the model checkpoints will be saved in the `exp` folder.
