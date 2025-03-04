function [ Lwp,  Wb] = laplacian_create( X_train, Y_label_f,S)
%LAPLACIAN_CREATE Summary of this function goes here
%   Detailed explanation goes here
        [train_num, ~] = size(X_train);
        
        Wwp=zeros(train_num);
        Wwp=double(Wwp);
        
        Wwn=zeros(train_num);
        Wwn=double(Wwn);
        
        Wb=zeros(train_num);
        Wb=double(Wb);
        
        for i=1:train_num
            for j=1:train_num
%                 fprintf('--> %d-%d \t  iter. \n', i,train_num);
                if (i~=j)
                    if (Y_label_f(i)==Y_label_f(j))
                        if (Y_label_f(i)==1)
                            Wwp(i,j)= S(i,j);
                        else
                            Wwn(i,j)= S(i,j);
                        end
                    else
                        Wb(i,j)=S(i,j);
                    end
                end
            end
        end
        
        Wwp = (Wwp + Wwp')/2;
        Wwn = (Wwn + Wwn')/2;
        Wb = (Wb + Wb')/2;
        
        Dwp=zeros(train_num);
        Dwp=double(Dwp);
        Dwp = diag(sum(Wwp,2));
        
        Dwn=zeros(train_num);
        Dwn=double(Dwn);
        Dwn = diag(sum(Wwn,2));
        
        Db=zeros(train_num);
        Db=double(Db);
        Db = diag(sum(Wb,2));
        
         
        
        Lwp=zeros(train_num);
        Lwp=double(Lwp);
        Lwp=Dwp-Wwp;
        
        Lwn=zeros(train_num);
        Lwn=double(Lwn);
        Lwn=Dwn-Wwn;
        
        Lb=zeros(train_num);
        Lb=double(Lb);
        Lb=Db-Wb;


end

