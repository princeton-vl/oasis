function varargout = edgesEvalClassDir( varargin )
%
% In addition to the outputs, edgesEvalDir() creates three files:
%  eval_bdry_img.txt - per image OIS results [imgId T R P F]
%  eval_bdry_thr.txt - per threshold ODS results [T R P F]
%  eval_bdry.txt     - complete results (*re-ordered* copy of output)
% These files are identical to the ones created by boundaryBench.
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get additional parameters
dfs={'resDir','REQ','occDir','REQ','foldDir', 'REQ', 'pDistr',{{'type','parfor'}}, ...
  'cleanup',0, 'thrs',99, 'maxDist',.0075, 'thin',1 };
p=getPrmDflt(varargin,dfs,1); resDir=p.resDir; occDir=p.occDir; foldDir=p.foldDir;
evalDir=[resDir '-eval/']; if(~exist(evalDir,'dir')), mkdir(evalDir); end
evaloccDir=[occDir '-eval/']; evalfoldDir=[foldDir '-eval/'];

% check if results already exist, if so load and return
%{
fNm = fullfile(evalDir,'eval_bdry.txt');
if(exist(fNm,'file')), R=dlmread(fNm); newR=mat2cell2(R,[1 8]);
  varargout=newR([4 3 2 1 7 6 5 8]);
  fprintf('ODS-F=%.3f OIS-F=%.3f ODS-P=%.3f OIS-P=%.3f ODS-R=%.3f OIS-R=%.3f AP=%.3f', R([4 7 3 6 2 5 8]));
  return;
end
%}

% perform evaluation on each image
assert(exist(resDir,'dir')==7); assert(exist(occDir,'dir')==7); assert(exist(foldDir,'dir')==7);
assert(exist(evaloccDir,'dir')==7); assert(exist(evalfoldDir,'dir')==7);
ids=dir(fullfile(occDir,'*.png'));
ids={ids.name}; n=length(ids);

res=cell(2,n); % occlusion / fold
for i=1:n, id=ids{i}(1:end-4);
  res{1,i}=fullfile(evaloccDir,[id '_ev1.txt']);
  res{2,i}=fullfile(evalfoldDir,[id '_ev1.txt']);
end

% collect evaluation results
I=dlmread(res{1,1}); T=I(:,1);
Z=zeros(numel(T),1); cntR=Z; sumR=Z; cntP=Z; sumP=Z;
oisCntR=0; oisSumR=0; oisCntP=0; oisSumP=0; scores=zeros(n,5);
for i=1:n
  % load image results and accumulate
  I_occ = dlmread(res{1,i});
  I_fold = dlmread(res{2,i});

  cntR1=I_occ(:,2)+I_fold(:,2); cntR=cntR+cntR1;
  sumR1=I_occ(:,3)+I_fold(:,3); sumR=sumR+sumR1;
  cntP1=I_occ(:,4)+I_fold(:,4); cntP=cntP+cntP1;
  sumP1=I_occ(:,5)+I_fold(:,5); sumP=sumP+sumP1;
  % compute OIS scores for image
  [R,P,F] = computeRPF(cntR1,sumR1,cntP1,sumP1); [~,k]=max(F);
  [oisR1,oisP1,oisF1,oisT1] = findBestRPF(T,R,P);
  scores(i,:) = [i oisT1 oisR1 oisP1 oisF1];
  % oisCnt/Sum will be used to compute dataset OIS scores
  if (sumR1(1) == 0) % This is necessary to avoid a bug when there is no positive instances in a gt image.
      bestk=length(sumR1);
  else
      [~,bestk]=max(F);
  end
  oisCntR=oisCntR+cntR1(bestk);
  oisSumR=oisSumR+sumR1(bestk);
  oisCntP=oisCntP+cntP1(bestk);
  oisSumP=oisSumP+sumP1(bestk);
end

% compute ODS R/P/F and OIS R/P/F
[R,P,F] = computeRPF(cntR,sumR,cntP,sumP);
[odsR,odsP,odsF,odsT] = findBestRPF(T,R,P);
%[odsR,odsP,odsF,odsT] = findBestDebugRPF(T,R,P); % No interpolation
[oisR,oisP,oisF] = computeRPF(oisCntR,oisSumR,oisCntP,oisSumP);
% compute AP/R50 (interpolating 100 values, has minor bug: should be /101)
if(0), R=[0; R; 1]; P=[1; P; 0]; F=[0; F; 0]; T=[1; T; 0]; end
[~,k]=unique(R); k=k(end:-1:1); R=R(k); P=P(k); T=T(k); F=F(k); AP=0;
if(numel(R)>1), AP=interp1(R,P,0:.01:1); AP=sum(AP(~isnan(AP)))/100; end
[~,o]=unique(P);
if(numel(R)>1), R50=interp1(P(o),R(o),max(P(o(1)),.5)); else, R50=R(o); end

% write results to disk
varargout = {odsF,odsP,odsR,odsT,oisF,oisP,oisR,AP,R50};
writeRes(evalDir,'eval_bdry_img.txt',scores);
writeRes(evalDir,'eval_bdry_thr.txt',[T R P F]);
writeRes(evalDir,'eval_bdry.txt',[varargout{[4 3 2 1 7 6 5 8]}]);
fprintf('ODS-Fscore=%.3f OIS-Fscore=%.3f ODS-Precision=%.3f OIS-Precision=%.3f ODS-Recall=%.3f OIS-Recall=%.3f AP=%.3f',odsF,oisF,odsP,oisP,odsR,oisR,AP);

function [R,P,F] = computeRPF(cntR,sumR,cntP,sumP)
% compute precision, recall and F measure given cnts and sums
R=cntR./max(eps,sumR); P=cntP./max(eps,sumP); F=2*P.*R./max(eps,P+R);
end

function [bstR,bstP,bstF,bstT] = findBestRPF(T,R,P)
% linearly interpolate to find best thr for optimizing F
if(numel(T)==1), bstT=T; bstR=R; bstP=P;
  bstF=2*P.*R./max(eps,P+R); return; end
A=linspace(0,1,100); B=1-A; bstF=-1;
for j = 2:numel(T)
  Rj=R(j).*A+R(j-1).*B; Pj=P(j).*A+P(j-1).*B; Tj=T(j).*A+T(j-1).*B;
  Fj=2.*Pj.*Rj./max(eps,Pj+Rj); [f,k]=max(Fj);
  if(f>bstF), bstT=Tj(k); bstR=Rj(k); bstP=Pj(k); bstF=f; end
end
end

function [bstR,bstP,bstF,bstT] = findBestDebugRPF(T,R,P)
% linearly interpolate to find best thr for optimizing F
if(numel(T)==1), bstT=T; bstR=R; bstP=P;
  bstF=2*P.*R./max(eps,P+R); return; end
A=linspace(0,1,100); B=1-A; bstF=-1;
for j = 2:numel(T)
  Rj=R(j); Pj=P(j); Tj=T(j);
  Fj=2.*Pj.*Rj./max(eps,Pj+Rj); [f,k]=max(Fj);
  if(f>bstF), bstT=Tj(k); bstR=Rj(k); bstP=Pj(k); bstF=f; end
end
end

function writeRes( alg, fNm, vals )
% write results to disk
k=size(vals,2); fNm=fullfile(alg,fNm); fid=fopen(fNm,'w');
if(fid==-1), error('Could not open file %s for writing.',fNm); end
frmt=repmat('%10g ',[1 k]); frmt=[frmt(1:end-1) '\n'];
fprintf(fid,frmt,vals'); fclose(fid);
end
end
