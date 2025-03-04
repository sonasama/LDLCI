function [target,gradient] = process(weights)

% 声明全局变量
global train_feature train_distribution alpha beta gama S

% 计算当前参数weights下的预测，其中modProb_posCent与modProb_negCent分别是对各个标记组的正类及负类的特征中心预测
[modProb,~] = predict(train_feature,weights);%modProbΪ��ѵ�����г˺�Ľ��

% 计算损失函数及各个正则项的�?�及梯度
[loss,grad_loss] = calcLoss(train_feature,train_distribution,modProb);
[normL21,grad_norm] = calcL21(weights);
[sloss,grad_sloss]=calicl(S,weights,train_feature);
[LCloss,LCloss_grad] = caljlc(weights,train_feature,train_distribution);
% �?终结�?
target = loss + alpha*normL21 + beta*LCloss+ gama*sloss;
gradient = grad_loss + alpha*grad_norm + beta*LCloss_grad + gama*grad_sloss;
end

