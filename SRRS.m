function [Z,E,P] = SRRS(X, label, para)
% The main function of Supervised Regularization based Robust Subspace (SRRS) Method.
%  Input: 
%      X: d*n data matrix
%  label: label vector
%   para: parameter
% 
% Output:
%      Z: coefficient matrix
%      E: error matrix
%      P: projection matrix
%
% Author: Sheng Li (shengli@ece.neu.edu)
% June 29, 2015

%%% Get parameters
lambda1 = para.lambda1;
lambda2 = para.lambda2;
eta = para.eta;
tol = para.tol;

%%% Get the number of classes
cn = length(unique(label));

%%% Get the size of data
[d, n] = size(X);

%%% Parameter setting in ALM
maxIter = 3e2;
rho = 1.3;
max_mu = 1e10;
mu = 6e-1; 
xtx = X'*X;
ni = n/cn;
Ht = (1/n)*ones(n,n);
Hb = (1/ni)*ones(ni,ni);
Hb = blkdiag(Hb,Hb);
for i = 1:ceil(log(cn)/log(2))
    Hb = blkdiag(Hb,Hb);
end
Hb = Hb(1:n,1:n);

lyA_1 = X'*X + 1*eye(n);
lyB = lambda1*((1+eta)*eye(n)+Hb*Ht-Hb-Ht*Ht);

Z = zeros(d, n);
E = sparse(d, n);
P = rand(d, cn);
Y1 = zeros(d,n);
Y2 = zeros(n,n);


%% The main loop of optimization
iter = 0;
while iter<maxIter
    iter = iter + 1;
    
    if iter == 1
        Z = 1;
    end
    paraP = [];
    paraP.Hb = Hb;
    paraP.Ht = Ht;
    paraP.lambda = lambda1;
    paraP.eta = eta;
    P = UpdateP(X*Z, paraP); 
    
    %%% Update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %%% Update Z
    kly = X'*(P*P')*X + 1e-4*eye(size(X,2));
    lyA = kly \ lyA_1;
    lyB = lyB/mu;
    lyC = - (kly \ (xtx-X'*E+J+(X'*Y1-Y2)/mu));
    Z = lyap(lyA,lyB,lyC);
    
    %%% Update E
    xmaz = X - X*Z;
    temp = xmaz + (Y1 / mu);
    E = solve_l1l2(temp, lambda2/mu);
    
    leq1 = xmaz - E;
    leq2 = Z - J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    if iter==1 || mod(iter,20)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopC=' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol 
        disp('SRRS done.');
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end