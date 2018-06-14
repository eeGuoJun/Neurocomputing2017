function [CoefMat,AnalyMat,R_Mat,obj_value] = TrainAL(Data,H_Mat,alpha,gamma,beta,T,iterative)
% This is the training function of SLC-ADL

ClassNum = size(H_Mat,1);% When the row of H is extended, the ClassNum is the real number of class*Magni.
DataSize = size(Data,2);
Dim = size(Data,1);
CoefMat = zeros(ClassNum,DataSize);
I_Mat_Y = eye(Dim,Dim);
I_Mat_R = eye(ClassNum,ClassNum);
obj_value = [];
ChangeR_total = [];
ChangeR_max = [];
CoefMat = H_Mat;
R_Mat = eye(ClassNum,ClassNum);
R{1} = R_Mat;

%% iteratively updating the AnalyMat, CoefMat, R
for i=1:iterative
    AnalyMat = CoefMat*Data'/(Data*Data'+beta*I_Mat_Y);
    norm_AnalyMat = normcol_equal( AnalyMat');	
    AnalyMat=norm_AnalyMat';
    R_Mat = CoefMat*H_Mat'/(H_Mat*H_Mat'+gamma*I_Mat_R);
    CoefMat = (AnalyMat*Data+alpha*R_Mat*H_Mat)/(1+alpha);
    CoefMat = SR_Cons(CoefMat,T);
    R{i+1} = R_Mat;
    ChangeR_total = [ChangeR_total norm(R{i+1}-R{i},'fro')^2];
    ChangeR_max = [ChangeR_max max(max(abs(R{i+1}-R{i})))];
    obj_value = [obj_value norm(CoefMat-AnalyMat*Data,'fro')^2 + ...
        alpha*norm(CoefMat-R_Mat*H_Mat,'fro')^2 + ...
        beta*norm(AnalyMat,'fro')^2 ];
end
fprintf('\nDone!');