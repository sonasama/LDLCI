clc;
clear;
clear all;

%add current file path and sub file path
%addpath(genpath(pwd));

%% parameter setting
para.rep = 5; % cross validation times
para.NIter = 20; % maximum number of iterations
para.lamda1 = 0.01;
para.lamda2 = 0.01;
para.lamda3 = 0.01;
para.lamda4= 0.01;
dataname='Recreation_data.mat'; 
% tic;
fprintf('--> begin to load dataset: %s. \n',dataname);
load(dataname);
if exist('train_data','var') == 1
    data = [train_data; test_data];
    % data = zscore(data); % feature normalization
    target = [train_target'; test_target'];
    clear train_data test_data train_target test_target
else
    disp('preprocessing the data set is required');
    
    return;
    
end
% append another feature (all equals to 1) to the data
[num_data,num_feature_origin]= size(data);
data(:, num_feature_origin + 1) = 1;
% transform 1/-1 => 1/0
ind = find(target == -1);
target(ind) = 0;
randorder = randperm(num_data);
PRO=zeros(8,para.rep);
for t = 1:para.rep
    % train test split
    fprintf('--> %s dataset train test split. \n',dataname);
    [X_train, Y_train, X_test, Y_test] = generateCVSet( data,target,randorder,t,para.rep );
    [train_num, feature_num] = size(X_train);
    relation = corrcoef(X_train');
    train_feature=X_train;
    global train_feature
%     %item = zeros(d,c);
   [r,c]=size(X_train);
    S1=rand(r,r);


     option = optimoptions(@fminunc,...
                        'Display','None','Algorithm','quasi-newton','MaxIterations',500000,'HessianApproximation','lbfgs',...
                        'FiniteDifferenceStepSize',1e-20,'SpecifyObjectiveGradient',true,'TolX',1e-12,'GradObj','on');
S1 = fminunc(@calSS, S1,option);%itemΪx
E=eye(r,r);
global S1


    [test_num, ~] = size(X_test);
    [~,label_dim] = size(Y_train);
    %Cost = Preprocess_Cost_Sensitive_ours(Y_train);%文中的C，控制计算每个样本的权重
    W = rand(feature_num, label_dim);%随机生成特征矩阵
    objs = zeros(para.NIter,1);
    obji = 1;
    P = Y_train;
    
    %% K-NN similarity matrix construct
    neighboor_K = label_dim + 1;
    Dis = pdist2(X_train , X_train , 'Euclidean' );
    [dumb,idx] = sort(Dis, 2); % sort each row
    [dumbP,idxP] = sort(relation, 2); % sort each row
    S = spalloc(train_num,train_num,20*train_num);%生成n*n的稀疏矩阵可以有20n个非零值的空间
    for nei_i = 1 : train_num
        S(nei_i,idx(nei_i,2:neighboor_K)) = 1;%S为训练样本的邻域为1
    end
    T=S1-E;
    
    obj_k = zeros(label_dim,1);
    
    for iter = 1 : para.NIter%开始迭代
        for k = 1 : label_dim%对每个标签
            %U = diag(Cost(:,k));% 用C的第k个标签构建一个对角矩阵spdiags(Cost(:,k),0, train_num, train_num);
            d = 0.5./sqrt(sum(W(:,k).*W(:,k), 2) + eps);%eps是为了防止为0，d的本质是欧式距离，源于2,1范数
            D = diag(d);
            W(:,k) = (X_train'*X_train+para.lamda3*D)\(X_train'*P(:,k));%公式11？
            Wwp=zeros(train_num);
            Wwp=double(Wwp);
            
            Wwn=zeros(train_num);
            Wwn=double(Wwn);
            
            Wb=zeros(train_num);
            Wb=double(Wb);
            
            for num_j=1:train_num
                for f_k=1:train_num
                    if (num_j~=f_k)
                        if relation(num_j,f_k)>0%标记空间上值相同
                            if S1(num_j,f_k)>0
                                Wwp(num_j,f_k)= 1;%得到正同类这里的S是训练样本的邻域
                            else
                                %Wwn(num_j,f_k)= S(num_j,f_k);%异同类同为0的
                            end
                        else
                            Wb(num_j,f_k)=1;%异类
                        end
                    end
                end
            end
            
            Wwp = (Wwp + Wwp')/2;
            Wwn = (Wwn + Wwn')/2;
            Wb = (Wb + Wb')/2;
            
            Dwp=zeros(train_num);
            Dwp=double(Dwp);
            Dwp = diag(sum(Wwp,2));%以该向量为主的对角线方阵
            
            Dwn=zeros(train_num);
            Dwn=double(Dwn);
            Dwn = diag(sum(Wwn,2));
            
            Db=zeros(train_num);
            Db=double(Db);
            Db = diag(sum(Wb,2));
            
            %L
            
            Lwp=zeros(train_num);
            Lwp=double(Lwp);
            Lwp=Dwp-Wwp;%所有非同正类的
            
            Lwn=zeros(train_num);
            Lwn=double(Lwn);
            Lwn=Dwn-Wwn;%所有非同异类的
            
            Lb=zeros(train_num);
            Lb=double(Lb);
            Lb=Db-Wb;%所有同类的F_b?
            
            P(:,k) = (Lwp+para.lamda1*Lwn+para.lamda2*Wb+para.lamda4*T*T')\(X_train*W(:,k)+X_train*W(:,k));%公式14"？更新D(U*U+Lwp+para.lamda1*Lwn+para.lamda2*Wb+W(:,k)'*W(:,k)+T*T')\(U*U*X_train*W(:,k)+X_train*W(:,k));
            A1 = find(P(:,k) > 1); P(A1,k) = 1;
            A0 = find(P(:,k) < 0); P(A0,k) = 0;%公式15
            obj_k(k) = 0.5*(norm(((X_train*W(:,k)-P(:,k)) ), 'fro'))^2+ 0.5*trace(P(:,k)'*(Lwp+para.lamda1*Lwn+para.lamda2*Wb)*P(:,k))+para.lamda3*sum(sqrt(sum(W(:,k).*W(:,k),2)+eps));%公式8
        end
        objs(iter) = sum(obj_k);
        cver = abs((objs(iter) - obji)/obji);
        obji = objs(iter);
        iter = iter + 1;
        if (cver < 10^-2 && iter > 2) , break, end
    end
    distribution = (softmax(P'))';
%     toc;
    % predict
    test_outputs = X_test*W;
    [l,m]=size(test_outputs);
    pre_labels=zeros(l,m);
    pre_labels(find(test_outputs<0.5))=-1;
    pre_labels(find(test_outputs>0.5))=1;
    
    ind_y = find(Y_test == 0);
    Y_test(ind_y) = -1;
    TT = EvaluationAll(pre_labels',test_outputs',Y_test');
    
    PRO(:,t) = TT;
    %remove current file and sub file path
    fprintf('--> %s dataset %d-th  times cross-validation end. \n',dataname,t);
end
Avg_Result_OURS_3(:,1)=mean(PRO,2);
Avg_Result_OURS_3(:,2)=std(PRO,1,2);
Avg_Result_OURS_3(:,1)=mean(PRO,2);
Avg_Result_OURS_3(:,2)=std(PRO,1,2);
AUC =  Avg_Result_OURS_3(1,1);
RankingLoss =  Avg_Result_OURS_3(2,1);
Coverage =Avg_Result_OURS_3(3,1);
Average_Precision =  Avg_Result_OURS_3(4,1);
HammingLoss =  Avg_Result_OURS_3(5,1);
Mic_F1 =  Avg_Result_OURS_3(6,1);
Mac_F1 =  Avg_Result_OURS_3(7,1);
One_erro =  Avg_Result_OURS_3(8,1);
% rmpath(genpath(pwd))