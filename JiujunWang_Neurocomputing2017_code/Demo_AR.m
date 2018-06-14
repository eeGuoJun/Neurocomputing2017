clear all
clc
warning off
addpath(genpath('.\files'));


%% %%%%%%%%%%%%%%%% SLC-ADL using Classifier 1
DataPath = '.\data\ARdata.mat';
load(DataPath);
fprintf('\n\n----- SLC-ADL using Classifier 1 -----');
% ------- parameter setting ------- 
Magni_H = 6;
T = 6;
s = 6;
gamma = 1e-6;
alpha = 50;    
beta = 1e-2;
iterative = 5;
% ------- the preprocessing of data ------- 
training_feats = normcol_equal(training_feats);	
testing_feats = normcol_equal(testing_feats);	
H_train =extend_H(H_train,Magni_H);
% ------- Algorithm trainig ------- 
tic
[CoefMat,AnalyMat,R_Mat,obj_value] = TrainAL(training_feats,H_train,alpha,gamma,beta,T,iterative);
TrTime = toc;
% ------- Algorithm testing ------- 
tic
PredictLabel = Classification1(testing_feats,AnalyMat,R_Mat,Magni_H,s);%The first classification scheme in our paper
[~,H_testLabel] = max(H_test);%H_testLabel is row number of the first 1 of H_test
Accuracy = sum(H_testLabel==PredictLabel)/size(H_test,2);
TtTime = toc;
fprintf('\nThe running time for training is %.03f ', TrTime);
fprintf('\nThe running time for testing is %.03f ', TtTime);
fprintf('\nClassification accuracy is %.03f \n', Accuracy);


%% %%%%%%%%%%%%%%%% SLC-ADL using Classifier 2 
DataPath = '.\data\ARdata.mat';
load(DataPath);
fprintf('\n\n----- SLC-ADL using Classifier 2 -----');
% ------- parameter setting ------- 
Magni_H = 6;
T = 6;
s = 600;
gamma = 1e-6;
alpha = 50;    
beta = 1e-1;
iterative = 5;
% the preprocessing of data ------- 
training_feats = normcol_equal(training_feats);	
testing_feats = normcol_equal(testing_feats);	
H_train =extend_H(H_train,Magni_H);
% ------- Algorithm trainig ------- 
tic
[CoefMat,AnalyMat,R_Mat,obj_value] = TrainAL(training_feats,H_train,alpha,gamma,beta,T,iterative);
TrTime = toc;
% ------- Algorithm testing ------- 
tic
[~,Accuracy] = NN_classify_HSR(AnalyMat,R_Mat,training_feats,testing_feats,H_test,H_train,Magni_H,s);%The second classification scheme in our paper
TtTime = toc;
fprintf('\nThe running time for training is %.03f', TrTime);
fprintf('\nThe running time for testing is %.03f ', TtTime);
fprintf('\nClassification accuracy is %.03f \n', Accuracy);


%% %%%%%%%%%%%%%%%% SLC-ADL using Classifier 3
DataPath = '.\data\ARdata.mat';
load(DataPath);
fprintf('\n\n----- SLC-ADL using Classifier 3 -----');
% ------- parameter setting ------- 
Magni_H = 6;
T = 6;
s = 6;
gamma = 1e-6;
alpha = 50;    
beta = 1e-2;
iterative = 5;
% ------- the preprocessing of data ------- 
training_feats = normcol_equal(training_feats);	
testing_feats = normcol_equal(testing_feats);	
H_train =extend_H(H_train,Magni_H);
% ------- Algorithm trainig ------- 
tic
[CoefMat,AnalyMat,R_Mat,obj_value] = TrainAL(training_feats,H_train,alpha,gamma,beta,T,iterative);
TrTime = toc;
% ------- Algorithm testing ------- 
tic
[PredictLabel] = Classification3(testing_feats,AnalyMat,R_Mat,gamma,Magni_H,s);%The third classification scheme in our paper
[~,H_testLabel] = max(H_test);%H_testLabel is row number of the first 1 of H_test
Accuracy = sum(H_testLabel==PredictLabel)/size(H_test,2);
TtTime = toc;
fprintf('\nThe running time for training is %.03f ', TrTime);
fprintf('\nThe running time for testing is %.03f ', TtTime);
fprintf('\nClassification accuracy is %.03f \n', Accuracy);