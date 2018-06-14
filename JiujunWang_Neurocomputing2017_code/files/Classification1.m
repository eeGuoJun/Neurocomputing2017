function LabelVector = Classification1(TestData,AnalyMat,R_Mat,Magni_H,s)
%argmin ||x-Rl||

DataSize = size(TestData,2);
Dim_H = size(R_Mat,1);
ClassNum = Dim_H/Magni_H;
H_Mat    = eye(ClassNum,ClassNum);
H_Mat    = extend_H(H_Mat,Magni_H);
I_Mat    = eye(Dim_H,Dim_H);
TestCoef = AnalyMat*TestData;
if s<Dim_H
    TestCoef = SR_Cons(TestCoef,s); % constrain the TestCoef to be sparse
end

D_judge  = EuDist2(TestCoef',(R_Mat*H_Mat)',0);
[~,LabelVector] = min(D_judge');