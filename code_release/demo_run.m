clear all
addpath(genpath('.'));


%% Load Data
FileName = 'bc_a8a';
load([FileName,'.mat'])
X = data;
Y = labels;
X = NormalizeData(X,2);
X = NormalizeData(X,1);

%% Main Run
for i = 1:10
    ID         = randperm((length(Y)));
    Predict{i} = run_OHGD(Y,X,ID);
    R.Acc{i}   = sum(Predict{i}==Y(ID)')/length(Y);
end




