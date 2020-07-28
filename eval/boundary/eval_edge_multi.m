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

% % Debug mode
% if exist('test_dir','var')
%   boundary_dir = strcat(test_dir, '/boundary');
%   class_dir = strcat(test_dir, '/class');
%   boundary_gt_dir = strcat(test_dir, '/boundary_eval_format');
%   occ_gt_dir = strcat(test_dir, '/occ_eval_format');
%   fold_gt_dir = strcat(test_dir, '/fold_eval_format');
%   mask_dir = strcat(test_dir, '/mask_eval_format');
% end

% Set variables if they are not specified
if ~exist('thrs','var')
  thrs = 99;
end
if ~exist('boundary_gt_dir','var')
  % boundary_gt_dir = '/home/wfchen/p-surfaces/nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold/boundary_eval_format';
  boundary_gt_dir = strcat(gt_data_folder, '/boundary_eval_format');
end
if ~exist('occ_gt_dir','var')
  % occ_gt_dir = '/home/wfchen/p-surfaces/nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold/occ_eval_format';
  occ_gt_dir = strcat(gt_data_folder, '/occ_eval_format');
end
if ~exist('fold_gt_dir','var')
  % fold_gt_dir = '/home/wfchen/p-surfaces/nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold/fold_eval_format';
  fold_gt_dir = strcat(gt_data_folder, '/fold_eval_format');
end
if ~exist('mask_dir','var')
  % mask_dir = '/home/wfchen/p-surfaces/nori_training_tmp/hed_xwjabc/data/OASIS_occ_fold/mask_eval_format';
  mask_dir = strcat(gt_data_folder, '/mask_eval_format');
end

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
    fprintf('%d\n', i);
    if exist(fullfile(boundary_nms_dir, boundary_nms_names{i}),'file')
      continue;
    end
    
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

    % 3. Mask out regions outside of the annotation ROIs
    if exist('mask_dir','var')
      matObj = matfile(fullfile(mask_dir, boundary_mat_names{i}));
      varlist = who(matObj);
      mask = matObj.(char(varlist));
      E = E .* single(mask);
    end

    % 4. Saving processed imgs
    imwrite(uint8(E*255),fullfile(boundary_nms_dir, boundary_nms_names{i}))
    % fprintf('Done saving to %s\n', fullfile(boundary_nms_dir, boundary_nms_names{i}));
end

% disp('Step 2: matching ...')
% resDir = fullfile(boundary_dir, 'nms');
% if exist('test_dir','var')
%     edgesEvalDir('resDir',resDir,'gtDir',boundary_gt_dir, 'thrs', thrs, 'thin', 0, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
% else
%     edgesEvalDir('resDir',resDir,'gtDir',boundary_gt_dir, 'thrs', thrs, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
% end

% disp('Step 3: evaluating ...')
% figure;
% edgesEvalPlot(resDir,'HED');

% Section 2. Classification evaluation
% - Preprocessing (NMS Process: formerly nms_process.m from HED repo).
% - Matching & calculating TP, FP, TN
% - Evaluating with ODS / OIS scores
disp('  ');
disp('Starting classfifcation evaluation.');
fprintf('Classification dir: %s\n', class_dir);

class_mat_dir = fullfile(class_dir, 'mat');
class_nms_dir = fullfile(class_dir, 'nms');
occ_nms_dir = fullfile(class_dir, 'occ_nms');
fold_nms_dir = fullfile(class_dir, 'fold_nms');
mkdir(class_nms_dir); mkdir(occ_nms_dir); mkdir(fold_nms_dir);

files = dir(class_mat_dir);
files = files(3:end,:);  % It means all files except ./.. are considered.

class_mat_names = cell(1,size(files, 1));
occ_nms_names = cell(1,size(files, 1));
fold_nms_names = cell(1,size(files, 1));
for i = 1:size(files, 1),
    class_mat_names{i} = files(i).name;
    occ_nms_names{i} = [files(i).name(1:end-4), '.png']; % identical to below
    fold_nms_names{i} = [files(i).name(1:end-4), '.png']; % identical to above
end

disp('Preprocessing ...')
for i = 1:size(class_mat_names,2),
    if exist(fullfile(fold_nms_dir, fold_nms_names{i}),'file')
      continue;
    end


    % 1. Loading mat files for classification results
    % Note that positive predictions are occlusions and fold otherwise
    matObj = matfile(fullfile(class_mat_dir, class_mat_names{i}));
    varlist = who(matObj);
    x = matObj.(char(varlist));



    % 2. Loading a preocessed image for boundary detection
    E = imread(fullfile(boundary_nms_dir, boundary_nms_names{i}));

    % 3. Generate probability maps one each for occlusion and fold
    % Mask out the NMS results with class labels
    occ_E = E; fold_E = E;
    occ_E(x<0.5)=0; fold_E(x>=0.5)=0;

    % 4. Saving processed occ / fold imgs
    imwrite(uint8(occ_E),fullfile(occ_nms_dir, occ_nms_names{i}));
    imwrite(uint8(fold_E),fullfile(fold_nms_dir, fold_nms_names{i}));

    % fprintf('Done saving to %s\n', fullfile(fold_nms_dir, fold_nms_names{i}) );
end

disp('Step 2: matching ...')
parpool('local',20);
if exist('test_dir','var')
    edgesMatchClassDir('resDir',occ_nms_dir,'gtDir',occ_gt_dir,'thrs', thrs, 'thin', 0, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
    edgesMatchClassDir('resDir',fold_nms_dir,'gtDir',fold_gt_dir,'thrs', thrs, 'thin', 0, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
else
    edgesMatchClassDir('resDir',occ_nms_dir,'gtDir',occ_gt_dir,'thrs', thrs, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
    edgesMatchClassDir('resDir',fold_nms_dir,'gtDir',fold_gt_dir,'thrs', thrs, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
end
%dbstop 70 in edgesEvalClassDir
disp('Step 3: evaluating ...')
edgesEvalClassDir('resDir',class_nms_dir,'occDir',occ_nms_dir,'foldDir', fold_nms_dir);
