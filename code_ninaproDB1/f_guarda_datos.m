function[]=f_guarda_datos(a_knn_RMS, a_svm_RMS, a_lda_RMS, a_rf_RMS, a_mlp_RMS, a_lstm_RMS, ...
    a_knn_MAV, a_svm_MAV, a_lda_MAV, a_rf_MAV, a_mlp_MAV, a_lstm_MAV, ...
    a_knn_VAR, a_svm_VAR, a_lda_VAR, a_rf_VAR, a_mlp_VAR, a_lstm_VAR, ...
    p_knn_RMS, r_knn_RMS, s_knn_RMS, f1_knn_RMS,...
    p_svm_RMS, r_svm_RMS, s_svm_RMS, f1_svm_RMS,...
    p_lda_RMS, r_lda_RMS, s_lda_RMS, f1_lda_RMS,...
    p_rf_RMS, r_rf_RMS, s_rf_RMS, f1_rf_RMS,...
    p_mlp_RMS, r_mlp_RMS, s_mlp_RMS, f1_mlp_RMS,...
    p_lstm_RMS, r_lstm_RMS, s_lstm_RMS, f1_lstm_RMS,...
    p_knn_MAV, r_knn_MAV, s_knn_MAV, f1_knn_MAV,...
    p_svm_MAV, r_svm_MAV, s_svm_MAV, f1_svm_MAV,...
    p_lda_MAV, r_lda_MAV, s_lda_MAV, f1_lda_MAV,...
    p_rf_MAV, r_rf_MAV, s_rf_MAV, f1_rf_MAV,...
    p_mlp_MAV, r_mlp_MAV, s_mlp_MAV, f1_mlp_MAV,...
    p_lstm_MAV, r_lstm_MAV, s_lstm_MAV, f1_lstm_MAV,...
    tt_knn_RMS, tt_svm_RMS, tt_lda_RMS, tt_rf_RMS, tt_nb_RMS, tt_mlp_RMS, tt_lstm_RMS, ...
    tt_knn_MAV, tt_svm_MAV, tt_lda_MAV, tt_rf_MAV, tt_nb_MAV, tt_mlp_MAV, tt_lstm_MAV, ...
    pt_knn_RMS, pt_svm_RMS, pt_lda_RMS, pt_rf_RMS, pt_nb_RMS, pt_mlp_RMS, pt_lstm_RMS, ...
    pt_knn_MAV, pt_svm_MAV, pt_lda_MAV, pt_rf_MAV, pt_nb_MAV, pt_mlp_MAV, pt_lstm_MAV,...
    mtx_mean_cm_mav_lda,mtx_mean_cm_mav_lstm,data_db1)

    save('results_DB1.mat');
end

%     save('a_knn_RMS.mat', 'a_knn_RMS');
%     save('a_svm_RMS.mat', 'a_svm_RMS');
%     save('a_lda_RMS.mat', 'a_lda_RMS');
%     save('a_rf_RMS.mat', 'a_rf_RMS');
%     save('a_nb_RMS.mat', 'a_nb_RMS');
%     save('a_mlp_RMS.mat', 'a_mlp_RMS');
%     save('a_lstm_RMS.mat', 'a_lstm_RMS');
% 
%     save('a_knn_MAV.mat', 'a_knn_MAV');
%     save('a_svm_MAV.mat', 'a_svm_MAV');
%     save('a_lda_MAV.mat', 'a_lda_MAV');
%     save('a_rf_MAV.mat', 'a_rf_MAV');
%     save('a_nb_MAV.mat', 'a_nb_MAV');
%     save('a_mlp_MAV.mat', 'a_mlp_MAV');
%     save('a_lstm_MAV.mat', 'a_lstm_MAV');
% 
%     save('a_knn_VAR.mat', 'a_knn_VAR');
%     save('a_svm_VAR.mat', 'a_svm_VAR');
%     save('a_lda_VAR.mat', 'a_lda_VAR');
%     save('a_rf_VAR.mat', 'a_rf_VAR');
%     save('a_nb_VAR.mat', 'a_nb_VAR');
%     save('a_mlp_VAR.mat', 'a_mlp_VAR');
%     save('a_lstm_VAR.mat', 'a_lstm_VAR');
% 
%     save('p_svm_RMS.mat','p_svm_RMS');
%     save('r_svm_RMS.mat','r_svm_RMS');
%     save('s_svm_RMS.mat','s_svm_RMS');
%     save('f1_svm_RMS.mat','f1_svm_RMS');
% 
%     save('p_lda_RMS.mat','p_lda_RMS');
%     save('r_lda_RMS.mat','r_lda_RMS');
%     save('s_lda_RMS.mat','s_lda_RMS');
%     save('f1_lda_RMS.mat','f1_lda_RMS');
% 
%     save('p_rf_RMS.mat','p_rf_RMS');
%     save('r_rf_RMS.mat','r_rf_RMS');
%     save('s_rf_RMS.mat','s_rf_RMS');
%     save('f1_rf_RMS.mat','f1_rf_RMS');
% 
%     save('p_mlp_RMS.mat','p_mlp_RMS');
%     save('r_mlp_RMS.mat','r_mlp_RMS');
%     save('s_mlp_RMS.mat','s_mlp_RMS');
%     save('f1_mlp_RMS.mat','f1_mlp_RMS');
% 
%     save('p_lstm_RMS.mat','p_lstm_RMS');
%     save('r_lstm_RMS.mat','r_lstm_RMS');
%     save('s_lstm_RMS.mat','s_lstm_RMS');
%     save('f1_lstm_RMS.mat','f1_lstm_RMS');
% 
%     save('p_svm_MAV.mat','p_svm_MAV');
%     save('r_svm_MAV.mat','r_svm_MAV');
%     save('s_svm_MAV.mat','s_svm_MAV');
%     save('f1_svm_MAV.mat','f1_svm_MAV');
% 
%     save('p_lda_MAV.mat','p_lda_MAV');
%     save('r_lda_MAV.mat','r_lda_MAV');
%     save('s_lda_MAV.mat','s_lda_MAV');
%     save('f1_lda_MAV.mat','f1_lda_MAV');
% 
%     save('p_rf_MAV.mat','p_rf_MAV');
%     save('r_rf_MAV.mat','r_rf_MAV');
%     save('s_rf_MAV.mat','s_rf_MAV');
%     save('f1_rf_MAV.mat','f1_rf_MAV');
% 
%     save('p_mlp_MAV.mat','p_mlp_MAV');
%     save('r_mlp_MAV.mat','r_mlp_MAV');
%     save('s_mlp_MAV.mat','s_mlp_MAV');
%     save('f1_mlp_MAV.mat','f1_mlp_MAV');
% 
%     save('p_lstm_MAV.mat','p_lstm_MAV');
%     save('r_lstm_MAV.mat','r_lstm_MAV');
%     save('s_lstm_MAV.mat','s_lstm_MAV');
%     save('f1_lstm_MAV.mat','f1_lstm_MAV');
% 
%     save('a_knn_VAR.mat', 'a_knn_VAR');
%     save('a_svm_VAR.mat', 'a_svm_VAR');
%     save('a_lda_VAR.mat', 'a_lda_VAR');
%     save('a_rf_VAR.mat', 'a_rf_VAR');
%     save('a_nb_VAR.mat', 'a_nb_VAR');
%     save('a_mlp_VAR.mat', 'a_mlp_VAR');
%     save('a_lstm_VAR.mat', 'a_lstm_VAR');
% 
%     save('tt_knn_RMS.mat','tt_knn_RMS');
%     save('tt_svm_RMS.mat','tt_svm_RMS');
%     save('tt_lda_RMS.mat','tt_lda_RMS');
%     save('tt_nb_RMS.mat','tt_nb_RMS');
%     save('tt_rf_RMS.mat','tt_rf_RMS');
%     save('tt_mlp_RMS.mat','tt_mlp_RMS');
%     save('tt_lstm_RMS.mat','tt_lstm_RMS');
% 
%     save('tt_knn_MAV.mat','tt_knn_MAV');
%     save('tt_svm_MAV.mat','tt_svm_MAV');
%     save('tt_lda_MAV.mat','tt_lda_MAV');
%     save('tt_nb_MAV.mat','tt_nb_MAV');
%     save('tt_rf_MAV.mat','tt_rf_MAV');
%     save('tt_mlp_MAV.mat','tt_mlp_MAV');
%     save('tt_lstm_MAV.mat','tt_lstm_MAV');
% 
%     save('pt_knn_RMS.mat','pt_knn_RMS');
%     save('pt_svm_RMS.mat','pt_svm_RMS');
%     save('pt_lda_RMS.mat','pt_lda_RMS');
%     save('pt_nb_RMS.mat','pt_nb_RMS');
%     save('pt_rf_RMS.mat','pt_rf_RMS');
%     save('pt_mlp_RMS.mat','pt_mlp_RMS');
%     save('pt_lstm_RMS.mat','pt_lstm_RMS');
% 
%     save('pt_knn_MAV.mat','pt_knn_MAV');
%     save('pt_svm_MAV.mat','pt_svm_MAV');
%     save('pt_lda_MAV.mat','pt_lda_MAV');
%     save('pt_nb_MAV.mat','pt_nb_MAV');
%     save('pt_rf_MAV.mat','pt_rf_MAV');
%     save('pt_mlp_MAV.mat','pt_mlp_MAV');
%     save('pt_lstm_MAV.mat','pt_lstm_MAV');
% 
% end