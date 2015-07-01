%%%%--Demo code of Supervised Regularization based Robust Subspace(SRRS)--%%
%%%%--Author: Sheng Li (shengli@ece.neu.edu)--%%
%%%%--June 30, 2015--%%

clear all;
close all;

%%% Load the COIL-20 dataset (20 objects) with 20% corruptions
load COIL_20_20c;
%%% Load the random training index
load COIL_20_10train_Idx;

%%% Feature normalization
fea = double(fea);
for i = 1:size(fea,1)
    fea(i,:) = fea(i,:)/norm(fea(i,:));
end

train_num = 10;
rate_all = [];

%%% Main loop
for loop = 1:10
    train = fea(trainIdx(loop,:),:);
    test = fea(testIdx(loop,:),:);
    
    gnd_train = gnd(trainIdx(loop,:));
    gnd_test = gnd(testIdx(loop,:));

    para = [];
    para.lambda1 = 0.001;
    para.lambda2 = 0.9;
    para.eta = 1.5;
    para.tol = 1e-8;
    [Z, E, P] = SRRS(train', gnd_train, para); 
    rate_tmp = [];
    for dim = 1:1:size(P,2)
        train_new = Z'*train * P(:,1:dim);
        test_new = test * P(:,1:dim);
        ypred = knnclassify(test_new,train_new,gnd_train',1);
        acc = sum(abs(ypred-gnd_test)<0.1)/length(ypred);
        rate_tmp = [rate_tmp; acc];
    end
    rate_all = [rate_all max(rate_tmp)];
end
m_right = mean(rate_all);
disp(['The average recognition rate is '  num2str(m_right)]);