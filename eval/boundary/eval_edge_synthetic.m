% Command to run.
% (echo "boundary_dir = '../output/epoch-x-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
% Argument necessary for evaluation
% boundary_dir = '../../nori_training_tmp/hed_xwjabc/exps/debug/boundary';
% class_dir = '../../nori_training_tmp/hed_xwjabc/exps/debug/class';
% boundary_gt_dir = '../../nori_training_tmp/hed_xwjabc/data/eval/boundary_eval_format';
% occ_gt_dir = '../../nori_training_tmp/hed_xwjabc/data/3SIW_occ/occ_eval_format';
% fold_gt_dir = '../../nori_training_tmp/hed_xwjabc/data/3SIW_fold/fold_eval_format';
% mask_dir = '../../nori_training_tmp/hed_xwjabc/data/3SIW_fold/mask_eval_format';
% thrs = 99
% test_dir = './unit_test/tests/test1'

% Debug mode
if exist('test_dir','var')
  boundary_dir = strcat(test_dir, '/boundary');
  class_dir = strcat(test_dir, '/class');
  boundary_gt_dir = strcat(test_dir, '/boundary_eval_format');
  occ_gt_dir = strcat(test_dir, '/occ_eval_format');
  fold_gt_dir = strcat(test_dir, '/fold_eval_format');
  mask_dir = strcat(test_dir, '/mask_eval_format');
end

% Set variables if they are not specified
if ~exist('thrs','var')
  thrs = 99;
end
if ~exist('boundary_gt_dir','var')
  boundary_gt_dir = '/n/fs/pvl/datasets/3SIW/boundary/mat';
end
if ~exist('occ_gt_dir','var')
  occ_gt_dir = '/n/fs/pvl/datasets/3SIW/occlusion/mat';
end
if ~exist('fold_gt_dir','var')
  fold_gt_dir = '/n/fs/pvl/datasets/3SIW/fold/mat';
end
%if ~exist('mask_dir','var')
%  mask_dir = '/n/fs/pvl/datasets/3SIW/mask/mat';
%end

% Data directory boundary_dir should be defined outside.
addpath(genpath('./edges'));
addpath(genpath('./toolbox.badacost.public'));

% Section 1. Boundary evaluation
% - Preprocessing (NMS Process: formerly nms_process.m from HED repo).
% - Matching & calculating TP, FP, TN
% - Evaluating with ODS / OIS scores
disp('Starting boundary evaluation.');
fprintf('Boundary dir: %s\n', boundary_dir);

boundary_mat_dir = fullfile(boundary_dir, 'mat');
boundary_nms_dir = fullfile(boundary_dir, 'nms');
mkdir(boundary_nms_dir);

files = dir(boundary_mat_dir);
files = files(3:end,:);  % It means all files except ./.. are considered.
boundary_mat_names = cell(1,size(files, 1));
boundary_nms_names = cell(1,size(files, 1));
for i = 1:size(files, 1),
    boundary_mat_names{i} = files(i).name;
    boundary_nms_names{i} = [files(i).name(1:end-4), '.png'];
end

disp('Step 1: preprocessing ...')
for i = 1:size(boundary_mat_names,2),
    % 1. Loading mat files
    matObj = matfile(fullfile(boundary_mat_dir, boundary_mat_names{i}));
    varlist = who(matObj);
    x = matObj.(char(varlist));

    % 2. NMS (Disable this step in debugging)
    if exist('test_dir','var')
        E=single(x);
    else
        E=convTri(single(x),1);
        [Ox,Oy]=gradient2(convTri(E,4));
        [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
        O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
        E=edgesNmsMex(E,O,1,5,1.01,4);
    end

    % 4. Saving processed imgs
    imwrite(uint8(E*255),fullfile(boundary_nms_dir, boundary_nms_names{i}))
end

disp('Step 2: matching ...')
resDir = fullfile(boundary_dir, 'nms');
if exist('test_dir','var')
    edgesEvalDir('resDir',resDir,'gtDir',boundary_gt_dir, 'thrs', thrs, 'thin', 0, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
else
    edgesEvalDir('resDir',resDir,'gtDir',boundary_gt_dir, 'thrs', thrs, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
end

disp('Step 3: evaluating ...')
%figure;
edgesEvalPlot(resDir,'HED'); % This has some visualization issue.

