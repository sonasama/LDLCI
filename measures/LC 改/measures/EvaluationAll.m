
function ResultAll = EvaluationAll(pre_labels,Outputs,test_target)
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)

    % first category using rnak
    % which including 1. rankingloss, 2. coverage, 3. average precision
    % second category using classifation results
    % which including 4. hamming loss, 5. macro f1, 6. micro f1 and 7. one erro
    ResultAll = zeros(8,1); 

    % first category
    
    RankingLoss = Ranking_loss(Outputs,test_target); % 1/-1
    Coverage = coverage(Outputs,test_target); %1/-1
    Average_Precision = Average_precision(Outputs,test_target); %1/-1
    
    
    
    % second  
    HammingLoss = Hamming_loss(pre_labels, test_target);
    Mic_F1 = MicroF1(pre_labels, test_target);
    Mac_F1 = MacroF1(pre_labels, test_target);
    One_erro = One_error(pre_labels,test_target);
    
    
    test_target(test_target == -1) = 0; 
    AUC = ak_auc_tp_fp_diffrent_ks(Outputs',test_target');
    
    ResultAll(1,1)    = AUC; 
 
    ResultAll(2,1)    = RankingLoss; 
    ResultAll(3,1)    = Coverage; 
    ResultAll(4,1)    = Average_Precision; 
    ResultAll(5,1)    = HammingLoss; 
    ResultAll(6,1)    = Mic_F1; 
    ResultAll(7,1)    = Mac_F1; 
    ResultAll(8,1)    = One_erro;
    
%     fprintf( '\nAUC = %f,RankingLoss = %f,Coverage = %f,Average_Precision = %f,HammingLoss = %f,Mic_F1 = %f,Mac_F1 = %f,One_erro = %f\n', AUC,RankingLoss,Coverage,Average_Precision,HammingLoss,Mic_F1,Mac_F1,One_erro);
%                 
    
 
end