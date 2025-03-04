function n_data=normalize(label)

n_data=zeros(size(label));
[n,l]=size(label);

for i=1:n
    me=mean(label(i,:));
    for j=1:l
        if label(i,j)>me
            n_data(i,j)=1;
        else
            n_data(i,j)=0;
        end
    end
end


end