function [loss,grad_loss] = calcLoss(train_feature,train_distribution,modProb)
%   计算损失函数及梯�?
%   此处显示详细说明

loss = -sum(sum(train_distribution.*log(modProb)));
grad_loss = train_feature'*(modProb - train_distribution);
end%KLɢ��
