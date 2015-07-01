function [P] = MMC2(X, label, lambda1, eta)
X = X';
[n, d] = size(X);
c_list = unique(label);
c_num = length(c_list);
ReducedDim = size(X,1) - c_num;
[eigPCA, ~] = pca(X);
Y = X*eigPCA(:,1:ReducedDim);

m_all = mean(Y,1);
m = zeros(c_num, size(Y,2));
Hw = zeros(n, size(Y,2));
for i = 1:c_num
    ind = label==c_list(i);
    m(i,:) = mean(Y(ind,:), 1);
    Hw(ind, :) = Y(ind,:) - repmat(m(i,:), nnz(ind), 1);
end

Hb = m - repmat(m_all, c_num, 1);

Sb = Hb' * Hb / c_num;
Sw = Hw' * Hw / n;

[eigV, eigD] = eig(lambda1*(Sb - 1*Sw + eta * (Y'*Y)));

% [eigV, eigD] = eig(Sw \ Sb);

[~, d_site] = sort(diag(eigD),'descend');
V = eigV(:,d_site(1:rank(Sb)));
P = eigPCA(:,1:ReducedDim) * V;

