function varargout = edgesMatchClassDir( varargin )
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get parameters
dfs={'resDir','REQ','gtDir','REQ','pDistr',{{'type','parfor'}}, ...
  'cleanup',0, 'thrs',99, 'maxDist',.0075, 'thin',1 };
p=getPrmDflt(varargin,dfs,1); resDir=p.resDir; gtDir=p.gtDir;
evalDir=[resDir '-eval/']; if(~exist(evalDir,'dir')), mkdir(evalDir); end

% perform matching on each image (this part can be very slow)
display('Start matching.');
assert(exist(resDir,'dir')==7); assert(exist(gtDir,'dir')==7);
ids=dir(fullfile(resDir,'*.png'));
ids={ids.name}; n=length(ids);
do=false(1,n); jobs=cell(1,n); res=cell(1,n);
for i=1:n,
  id=ids{i}(1:end-4); %TODO put semi-colon (Noriyuki)
  res{i}=fullfile(evalDir,[id '_ev1.txt']);
  do(i)=~exist(res{i},'file'); % check if results already exist, if so load and return
  im1=fullfile(resDir,[id '.png']); gt1=fullfile(gtDir,[id '.mat']);
  jobs{i}={im1,gt1,'out',res{i},'thrs',p.thrs,'maxDist',p.maxDist,...
    'thin',p.thin}; if(0), edgesEvalImg(jobs{i}{:}); end
end
fevalDistr('edgesEvalImg',jobs(do),p.pDistr{:});
display('Done matching.');
