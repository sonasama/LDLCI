% 鐗瑰緛閫夋嫨娴佺▼

dataSetsName = ...
{
    'Yeast_alpha',
};
l=1;

alphas = 0.0001;
betas =0.001;
gamas =  0.001;
dName = dataSetsName{l};
fprintf('[%s]\n',dName);

load([dName,'.mat']);
times = 6;
fold = 5;
%获取特征数大小
[num_sample, ~] = size(features);
for itrator=1:times
    %5折交叉验证，crossvalind该命令返回一个对于N个观察样本的K个fold，2500行1到5的数据
    indices = crossvalind('Kfold', num_sample, fold);
    for rep=1:fold
        
        testIdx = find(indices == rep);
        %对于向量A,向量B，C=setdiff(A,B)函数返回在向量A中却不在向量B中的元素，
        %并且C中不包含重复元素，并且从小到大排序，为了确定除测试集外的训练集
        trainIdx = setdiff(find(indices),testIdx);
        %测试集和训练集
        test_feature = features(testIdx,:);
        test_distribution = labels(testIdx,:);
          
        train_feature = features(trainIdx,:);
        train_distribution = labels(trainIdx,:);
        %返回 A 的相关系数的矩阵，其中 A 的列表示随机变量，行表示观测值。
        relation = corrcoef(train_distribution);%矩阵相关系数
        %tic记录matlab命令执行时间       
       



train_feature= double(train_feature);
train_distribution = double(train_distribution);
global train_feature train_distribution alpha beta gama

alpha = alphas(l);  beta = betas(l); gama = gamas(l); %gama样本之间相关性，beta标记之间相关性

% 鏍囪鍏宠仈
% 4.basic variable
d = size(features,2);
c = size(labels,2);
% 5.training
item = zeros(d,c);
[r,c]=size(train_feature);
S=rand(r,r);


option = optimoptions(@fminunc,...
                        'Display','None','Algorithm','quasi-newton','MaxIterations',500000,'HessianApproximation','lbfgs',...
                        'FiniteDifferenceStepSize',1e-20,'SpecifyObjectiveGradient',true,'TolX',1e-12,'GradObj','on');
S = fminunc(@calSS, S,option);%itemΪx
global S
options = optimoptions(@fminunc,...
                        'Display','None','Algorithm','quasi-newton','MaxIterations',500000,...
                        'FiniteDifferenceStepSize',1e-20,'SpecifyObjectiveGradient',true,'TolX',1e-12,'GradObj','on');
weights = fminunc(@process, item,options);%item为w
%[modProb] = LSPredict(weights, features);结果过于差了
[modProb,sumProb] = predict1(test_feature,weights);
 distance = computeMeasures(modProb,  test_distribution);
 dis(itrator,:)=distance;
[~,index] = sort(sqrt(sum(weights.*weights,2)),'descend');
    end
end
mean=zeros(1,13);
    for j=1:13
         for i=1:5

            mean(j)=mean(j)+mea(i,j);
        end
    end
    mean=mean/5;
   t=zeros(1,13); 
 for j=1:13

for i=1:5


t(j)=std(mea(:,j));
end
 end
% 6.validate the performance of feature subsets
%metric = tenFoldResult(features,labels,index);
%metric = round(metric,4);