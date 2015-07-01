function [P] = UpdateP(X, paraP)

Hb = paraP.Hb;
Ht = paraP.Ht;
lambda = paraP.lambda;
eta = paraP.eta;

X = X';
ReducedDim = size(X,1) - 1;
[eigPCA, ~] = pca(X);
Y = X*eigPCA(:,1:ReducedDim);

Zw = (eye(size(Ht)) - Ht) * Y;
Zb = (Hb - Ht) * Y;

[eigV, eigD] = eig(lambda*(Zb'*Zb - 5*(Zw'*Zw) - (eta) * (Y'*Y)));

[~, d_site] = sort(diag(eigD),'ascend');
V = eigV(:,d_site(1:rank(Zw'*Zw)));
P = eigPCA(:,1:ReducedDim) * V;

