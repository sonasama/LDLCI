function [Loss,grad_Loss] = calSS(S)
%���������½����������������
%   �˴���ʾ��ϸ˵��
global train_feature

Loss=0.5*sum(sum(power(S*train_feature-train_feature,2)));
grad_Loss = (S*train_feature-train_feature)*train_feature';
end

