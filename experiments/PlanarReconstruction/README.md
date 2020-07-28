# PlanarReconstruction

## Data Prep

```bash
PATH_TO_OASIS_TEST_AND_TRAINVAL=XXX
ln -s $PATH_TO_OASIS_TEST_AND_TRAINVAL OASIS

mkdir data
ln -s $PATH_TO_OASIS_TEST_AND_TRAINVAL/OASIS_trainval data/OASIS_trainval
ln -s $PATH_TO_OASIS_TEST_AND_TRAINVAL/OASIS_test data/OASIS_test
```

## Evaluation

To reproduce the results presented in the paper, e.g. the PlanarReconstruction trained on ScanNet + OASIS, first download the pretrained models, then run:

```bash
PR_OUT_PATH='planar'
mkdir -p $PR_OUT_PATH
mkdir -p $PR_OUT_PATH/seg
mkdir -p $PR_OUT_PATH/mask
MODEL_PATH='experiments/SN_plane_lr0.0001/2/checkpoints/network_epoch_11.pt'

TEST_CSV=OASIS_test.csv

##################### make predictions
python predict.py eval with resume_dir=$MODEL_PATH dataset_csv=data/OASIS_test/$TEST_CSV image_path=data/OASIS_test/image output_path=$PR_OUT_PATH

tar -czf planar.tar.gz planar
```

Then submit planar.tar.gz the results to the [OASIS website](https://pvl.cs.princeton.edu/OASIS) to evaluate on the test set.

Below shows an example of running on the validation set.

```bash
PR_OUT_PATH='SN_plane_lr0.0001_val'
mkdir -p $PR_OUT_PATH
mkdir -p $PR_OUT_PATH/seg
mkdir -p $PR_OUT_PATH/mask
MODEL_PATH='experiments/SN_plane_lr0.0001/1/checkpoints/network_epoch_14.pt'
VAL_CSV=OASIS_val.csv

##################### make predictions
python predict.py eval with resume_dir=$MODEL_PATH dataset_csv=data/OASIS_trainval/$VAL_CSV image_path=data/OASIS_trainval/image output_path=$PR_OUT_PATH

##################### generate test_list.txt
python instance_gen.py --gt_root_folder data/OASIS_trainval --dataset_csv $VAL_CSV --pred_folder $PR_OUT_PATH

##################### switch to eval/inst_segmentation and evaluate
cd ../../eval/inst_segmentation
python evalInstSeg_nips.py --gt_list_txt ../../experiments/PlanarReconstruction/$PR_OUT_PATH/test_list.txt --pred_path ../../experiments/PlanarReconstruction/$PR_OUT_PATH/seg/ --gt_suffix .png
```


## Training

1. Prepare the training data:

```bash
# convert the csv file `train_csv` into `output_dir/train.txt`, and covert the data under `dataset_path` into the proper format and store in `output_dir`.
OASIS_PROCESSED_DATA_FOLDER=XXX
python data_tools/convert_OASIS.py --data_type=train --dataset_path=OASIS/OASIS_trainval/ --train_csv OASIS/OASIS_trainval/OASIS_train.csv --output_dir=$OASIS_PROCESSED_DATA_FOLDER
```

2. Download the model pretrained on ScanNet [here](https://drive.google.com/file/d/1Aa1Jb0CGpiYXKHeTwpXAwcwu_yEqdkte/view).

3. To replicate the models used in the paper, run:

```bash
# Train on OASIS alone
python main.py train with dataset.root_dir=$OASIS_PROCESSED_DATA_FOLDER solver.lr=0.0001 exp_name=plane_lr0.0001

# Train on ScanNet + OASIS
python main.py train with dataset.root_dir=$OASIS_PROCESSED_DATA_FOLDER solver.lr=0.0001 exp_name=SN_plane_lr0.0001 resume_dir=pretrained.pt
```


## Acknowledgement

The code is based on the implementation from <https://github.com/svip-lab/PlanarReconstruction>.