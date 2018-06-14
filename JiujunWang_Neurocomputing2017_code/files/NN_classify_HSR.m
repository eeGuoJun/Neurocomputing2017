function [prediction,accuracy] = NN_classify_HSR(P,R,TrainData,TtData,H_test,H_train,Magni_H,s)%TrainData,
% function£ºclassify the testing data
% input£º
%       P           -the learned analysis dictionary
%       TrainData   -training data£¨each column is a sample£©
%       TtData      -testing data£¨each column is a sample£©
%       H_test      -the label matrix of testing data
%       H_train     -the label matrix of training data
% output£º
%       prediction  -the predicted label of testing data
%       accuracy    -classification accuracy


%% encoding
Dim_H = size(R,2);
TestCoef = P*TtData;
TrainCoef = P*TrainData;
if s<Dim_H
TestCoef     = SR_Cons(TestCoef,s);
TrainCoef     = SR_Cons(TrainCoef,s);
end
TestData = (R'*R+(1e-6)*eye(size(R)))\(TestCoef); %The label matrix of testing data obtained by coding coefficients
clear TtData TestCoef
TrnData = (R'*R+(1e-6)*eye(size(R)))\(TrainCoef); %The label matrix of trainding data obtained by coding coefficients
clear TrainData P TrainCoef

%% classification process
TrnData = TrnData'; %each row represents a sample
[~,TrnLabel] = max(H_train);
TrnLabel = ((TrnLabel-1)/Magni_H+1)'; %obtain a column vector indicating the class label of each sample
clear H_train
TestData = TestData'; %each row represents a sample
[~,TestLabel] = max(H_test);
TestLabel = TestLabel';
clear H_test
prediction = knnclassify(TestData,TrnData,TrnLabel,1,'euclidean','nearest');
matchVec = find(prediction==TestLabel);
accuracy = length(matchVec)/length(TestLabel);%calculating the accuracy