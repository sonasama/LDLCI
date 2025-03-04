function [Loss,grad_Loss] = calSS(S1)
%计算利用下降法计算样本相关性
%   此处显示详细说明
global train_feature

Loss=0.5*sum(sum(power(S1*train_feature-train_feature,2)));
grad_Loss = (S1*train_feature-train_feature)*train_feature';
end

