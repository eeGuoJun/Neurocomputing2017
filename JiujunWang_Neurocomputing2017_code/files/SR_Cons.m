function X = SR_Cons(Y,T)
% function£ºthe hard thresholding of input matrix
%   min_X  ||Y-X||_F^2    s.t. ||X||_0 = T
% input£º
%       Y       -a matrix: Y
%		T		-each column ||X_(i)||_0 = T
% output£º
%       X       -a matrix: X


X = Y;
Yabs = abs(Y);
[matYabs,~] = sort(Yabs,1,'descend');
med = matYabs(T+1,:);
Yabs = Yabs - repmat(med,size(Y,1),1);
X(Yabs<=0) = 0;