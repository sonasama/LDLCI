function [Loss,grad_Loss] = calSS(S1)
%���������½����������������
%   �˴���ʾ��ϸ˵��
global train_feature

Loss=0.5*sum(sum(power(S1*train_feature-train_feature,2)));
grad_Loss = (S1*train_feature-train_feature)*train_feature';
end

