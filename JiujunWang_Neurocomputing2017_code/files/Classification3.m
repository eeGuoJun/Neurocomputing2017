function PredictLabel = Classification3(TestData,AnalyMat,R_Mat,lambda,Magni_H,s )
% The third classification scheme in our paper

DataSize = size(TestData,2);
Dim_H = size(R_Mat,2);
ClassNum = Dim_H/Magni_H;
H_Mat    = eye(ClassNum,ClassNum);
H_Mat    = extend_H(H_Mat,Magni_H);
I_Mat    = eye(size(R_Mat));
TestCoef = AnalyMat*TestData;
if s<Dim_H
TestCoef = SR_Cons(TestCoef,s);
end

PredictH = (R_Mat'*R_Mat+lambda*I_Mat)\(R_Mat'*TestCoef);
D = EuDist2(PredictH',H_Mat',0);%D is the Euclidean distance of the corresponding column of PredictH and H_test
[~,PredictLabel] = min(D');