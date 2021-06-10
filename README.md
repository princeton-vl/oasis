# OASIS: A Large-Scale Dataset for Single Image 3D in the Wild

![OASIS](teaser.gif)

This repository contains the code for the following paper:

    OASIS: A Large-Scale Dataset for Single Image 3D in the Wild,
    Weifeng Chen, Shengyi Qian, David Fan, Noriyuki Kojima, Max Hamilton, Jia Deng
    Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Please check the [project site](https://pvl.cs.princeton.edu/OASIS) for more details.


## Installation

The code has been tested on python 3.7, cuda 10.0, pytorch 1.1.0, gcc 8.4.0

```bash
conda create --name oasis python=3.7
conda activate oasis

conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch  
conda install opencv==3.4.2 h5py scipy pillow==6.1.0 scikit-learn
pip install sacred easydict pyyaml imageio==2.6.0 tb-nightly future tqdm
```

## Data Preparation

Please go to the [download page](https://oasis.cs.princeton.edu/download) and download all the images and annotations. Then untar:

```bash
mkdir OASIS
tar -xzf OASIS_images_v1.tar.gz  -C OASIS
tar -xzf OASIS_trainval_annotations_v1.tar.gz -C OASIS
```

The folder tree after these steps should look like:

```bash
OASIS
    - LICENSE
    - OASIS_trainval
        - image
        - meta
        - OASIS_train.csv
        - OASIS_val.csv
        - depth
        - normal
        - fold
        - occlusion
        - mask
        - DIW_style_rel_depth
        - segmentation
            - planar_instance
            - continuous_instance
    - OASIS_test
        - image
        - meta
        - OASIS_test.csv    
```

## Submitting to the OASIS Benchmark

To submit your predictions to the OASIS benchmark, store your predictions for all test images in a directory with the following format:
```
<parent-dir>/
    - *****.npy
    - *****.npy
    ...
    - *****.npy
```

Where `<parent-dir>` is one of `depth`, `occfold`, `normals` or `planar`, depending on which benchmark you are submitting to. Then run the upload_to_benchmark.py python script to submit your results to the leaderboard. This can take upwards of an hour on slower internet connections.

**Example:**
```
python upload_to_benchmark.py --task normal_bench \ 
--password 9e67a7866dtf484748fcaf07fh5724s4etc7b94c --public \
--email firstname@princeton.edu --submission_name Hourglass --affiliation Princeton
```

**Usage:**
```
positional arguments:
  submission_directory  The directory containing .npy files to tar and submit.

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           one of ['normal_bench', 'depth_bench',
                        'occfold_bench', 'planar_bench'].
  --affiliation AFFILIATION
                        Your Affiliation (will not be publicly displayed).
  --publication_title PUBLICATION_TITLE
                        Publication Title.
  --publication_url PUBLICATION_URL
                        Link to Publication.
  --authors AUTHORS     Authors.
  --submission_name SUBMISSION_NAME
                        Submission Name (The name that will appear on the
                        leaderboard).
  --email EMAIL         Email account entered when receiving a password for
                        OASIS.
  --password PASSWORD   OASIS account password. Requested via the OASIS login
                        page. Valid for four hours.
  --public              Make the submission public.
  --skip_taring         Assume the submission is already tarred into the temporary
                        directory.
```

## Experiments

The `experiment` folder contains code to reproduce the results for the following experiments:

* Depth Estimation
* Surface Normal Estimation
* Fold and Occlusion Boundary Detection
* Planar Instance Segmentation

Please refer to the README files under each folder for instructions on how to run the code.

To run on pretrained models, please first download the pretrained models [experiments.tar.gz](https://drive.google.com/file/d/1XE--nVIUEROud5YwNRUuvqJH3I_cR9kI/view?usp=sharing), and `tar -xzf experiments.tar.gz`.
