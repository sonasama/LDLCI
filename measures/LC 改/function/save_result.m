function save_result_ours(score, y, dataset, param1, param2, param3)
    %p = round(score);
    [rowP,colP] = size(score);
    p = zeros(rowP,colP);
    p = score > 0.5;
    temp1 = sum(sum(p));
    
	[micro_f1, macro_f1] = f1_score(p, y);
    
    [rowOfp,colOfp] = size(p);
    [rowOfy,colOfy] = size(y);
    
    p1 = zeros(rowOfp,colOfp);
    y1 = zeros(rowOfy,colOfy);
    
    
    for i = 1:rowOfp
        for j = 1:colOfp
            if  p(i,j)>0
                p1(i,j) = 1;
            else
                p1(i,j) = -1;
            end
        end
    end
    
    for i = 1:rowOfy
        for j = 1:colOfy
            if  y(i,j)>0
                y1(i,j) = 1;
            else
                y1(i,j) = -1;
            end
        end
    end    
    
    HammingLoss=Hamming_loss(p',y');
    RankingLoss=Ranking_loss(score',y');
    Average_Precision=Average_precision(score',y');
    Coverage=coverage(score',y');
    OneError=One_error(score',y');
	fprintf('micro_f1 = %f, macro_f1 = %f,HammingLoss=%f,RankingLoss=%f,Average_Precision=%f,Coverage=%f,OneError=%f,',micro_f1, macro_f1,...
        HammingLoss,RankingLoss,Average_Precision,Coverage,OneError);
    %fprintf('micro_f1 = %f, macro_f1 = %f\n',micro_f1, macro_f1);
        
	fid = fopen([dataset, '_result_OURS.csv'], 'a');
    fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', param1, param2, param3,micro_f1, macro_f1,...
        HammingLoss,RankingLoss,Average_Precision,Coverage,OneError);
    %fprintf(fid, '%f,%f,%f,%f,%f,%f,%f\n', param1, param2, param3,param4,param5,micro_f1, macro_f1);

    fclose(fid);
end
