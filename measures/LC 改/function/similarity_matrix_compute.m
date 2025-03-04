function [ S ] = similarity_matrix_compute( X_train, Y_train )
%SIMILARITY_MATRIX_COMPUTE Summary of this function goes here
%   Detailed explanation goes here

%% K-NN similarity matrix construct
[~,label_dim] = size(Y_train);
[train_num, ~] = size(X_train);

neighboor_K = label_dim + 1;

Dis = pdist2(X_train , X_train , 'Euclidean' );
[~,idx] = sort(Dis, 2); % sort each row
S = spalloc(train_num,train_num,20*train_num);
for nei_i = 1 : train_num
    S(nei_i,idx(nei_i,2:neighboor_K)) = 1;
end


end

