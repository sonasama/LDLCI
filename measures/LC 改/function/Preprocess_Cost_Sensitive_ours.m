function [ C ] = Preprocess_Cost_Sensitive_ours( Y )
%UNTITLED Summary of this function goes here
%   Input : Y = [num_instance, num_label]
%   Output: C = [num_instance, num_label]




% ind = find(Y == 0);
% Y(ind) = -1;
% 
% 
% [num_instance, num_label] = size(Y);
% C = zeros(num_label, num_label);
% sum_tmp = sum(Y);
% pos_num = (sum_tmp + num_instance) / 2;
% neg_num = num_instance - pos_num;
% for i = 1: num_label
%     for j = 1: num_label
%         if (i==j)
%             C(i, j) = neg_num(j) / pos_num(j);%
%         else
%             C(i, j) = 1;
%         end
%     end
% end
% beta = 0.5;
% C = C.^(beta);

ind = find(Y == 0);
Y(ind) = -1;%恢复标记空间为-1


[num_instance, num_label] = size(Y);
C = zeros(num_instance, num_label);%生成与标记空间相同的0矩阵
sum_tmp = sum(Y);%每个标签的-1个数减1的个数
pos_num = (sum_tmp + num_instance) / 2;%正样本个数
neg_num = num_instance - pos_num;%负样本的个数
for i = 1: num_instance
    for j = 1: num_label
        if Y(i, j) == 1
            C(i, j) = neg_num(j) / pos_num(j);%如果是正的则
        else
            C(i, j) = 1;
        end
    end
end
beta = 0.5;
C = C.^(beta);



end