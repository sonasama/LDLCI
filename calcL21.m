function [normL21,grad_norm,D] = calcL21(weights)
%   计算�?疏项及梯�?
%   此处显示详细说明
d = size(weights,1);
D = eye(d)./(2*sqrt(weights*weights')+1e-5);%Dii
normL21 = 2*trace(weights'*D*weights);
grad_norm = 2*(D*weights);
% grad_norm = -grad_norm;
end

