%% 5 de Septiembre 2025
%Directorio: C:\Users\Marco\Documents\MATLAB\EMG\Ninapro\DB3\sEMG_ninapro_db3_test_estadistic_v2

% incluye PRUEBA ESTADÍSTICA...Friedman + post-hoc de Nemenyi. 
% codigo para analizar la BDD Ninapro DB3 (con amputados)
% fs=2 KHz. duracción de movimiento= 5seg + 3 seg de descanso.
% Emg (12 columnas): 
% columnas 1-8 corresponden a los electrodos distribuidos equitativamente alrededor del antebrazo a la altura de la articulación radiohumeral. 
% columnas 9 y 10 contienen señales del punto de actividad principal de los músculos flexor y extensor superficial de los dedos, 
% columnas 11 y 12 contienen señales del punto de actividad principal de los músculos bíceps braquial y tríceps braquial.
clc; clear; close all; 
rng(42);  % Semilla fija para reproducibilidad
%% Directorio Archivos
base_folder = 'C:\Users\Marco\Documents\MATLAB\EMG\Ninapro\DB3';
sujetos = 11;
gestos = 17;
repeticiones = 6;
electrodos=12;
f_original = 2000;
f_nueva = 100;
factor = f_original / f_nueva;  % Factor de decimación = 10
%Procesamiento Wavelet
waveletName = 'db4'; numLevels = 1;
all_data = cell(sujetos, 1);
%% 1. Carga optimizada de datos
for u = 1:sujetos
    user_folder = fullfile(base_folder, sprintf('s%d_0', u),sprintf('DB3_s%d',u));
    file_path = fullfile(user_folder, sprintf('S%d_E1_A1.mat', u));
    % Carga selectiva de variables
    loaded_data = load(file_path, 'emg', 'stimulus');
    % Almacenamiento directo sin copias innecesarias
    all_data{u} = struct('emg', double(loaded_data.emg), 'stimulus', single(loaded_data.stimulus));
end
clear user_folder file_path loaded_data
%% 2. Procesamiento de etiquetas optimizado
rng_etiquetasUser = struct();  % Pre-inicialización de estructura
for i=1:sujetos
    stimulus=all_data{i}.stimulus; % etiqueta de movimiento
    [rng_etiquetas]=f_rangos_etiquetas(stimulus);
    [rng_etiquetas]=f_diferencias_rangos(rng_etiquetas,gestos);
    rng_etiquetasUser.(sprintf('U%d',i))=rng_etiquetas;
end
clear stimulus

min_diff=f_minima_diferencia(rng_etiquetasUser,sujetos,gestos); clear rng_etiquetas
%tic
data_ninapro_db3=f_resample_ninaprodb3(all_data,rng_etiquetasUser,min_diff,sujetos,gestos,repeticiones,electrodos,factor);
%toc
[n_sujetos, n_gestos, n_repeticiones, n_muestras, n_electrodos] = size(data_ninapro_db3); %obtengo dimensiones de matriz
features_list={'RMS','MAV','VAR','WL','HIST','CC','WAV'};
% Preasignar la celda data_users con el número de sujetos
data_users = cell(sujetos, 1);  % Pre-asignar

% Opcional: Iniciar pool paralelo
%if isempty(gcp('nocreate')), parpool; end
%delete(gcp);

mtx_mean_cm_mav_lda=zeros(n_gestos,n_gestos,n_sujetos);
mtx_mean_cm_mav_lstm=zeros(n_gestos,n_gestos,n_sujetos);

for i = 1:sujetos
    fprintf('\n\nDB3 - Sujeto: %d', i);
    emg = reshape(data_ninapro_db3(i, :, :, :, :),[n_gestos, n_repeticiones, n_muestras, n_electrodos]);
    % Extracción y balanceo de características
    balanced_features = struct();
    % Extraer características RMS, MAV, VAR.
    [signal_rms,signal_mav,signal_var] = process_signal(emg,gestos,repeticiones,electrodos);    
    balanced_features.RMS = f_balancear_mtx(signal_rms,gestos);
    balanced_features.MAV = f_balancear_mtx(signal_mav,gestos);

    [cm_promedio_mav_lda]=f_clasific_MAV_lda(balanced_features.MAV);
    mtx_mean_cm_mav_lda(:,:,i)=cm_promedio_mav_lda;

    [cm_promedio_mav_lstm]=f_clasific_MAV_lstm(balanced_features.MAV);
    mtx_mean_cm_mav_lstm(:,:,i)=cm_promedio_mav_lstm;

    balanced_features.VAR = f_balancear_mtx(signal_var,gestos);
    [signal_wl]=extraer_signal_WL(emg,gestos,repeticiones,electrodos);
    balanced_features.WL = f_balancear_mtx(signal_wl,gestos);
    [signal_HIST]=HIST_features(emg,gestos,repeticiones,electrodos);
    balanced_features.HIST = f_balancear_mtx(signal_HIST,gestos);
    [signal_CC]=CC_features(emg,gestos,repeticiones,electrodos);
    balanced_features.CC = f_balancear_mtx(signal_CC,gestos);
    [signal_WAV]=DWT_features(emg,waveletName,numLevels,gestos,repeticiones,electrodos);
    balanced_features.WAV = f_balancear_mtx(signal_WAV,gestos);
  
    % Clasificación de todas las características
    %data_MAV=balanced_features.MAV; %% PARA GENERAR LA GRAFICA DE MATRIZ DE CONFUSIÓN
    classifiers = {'knn', 'svm', 'lda', 'rf', 'mlp', 'lstm'};
    results = struct();
    
    for feat_idx = 1:length(features_list)
        feat_name = features_list{feat_idx};
        fprintf('\n\n%s', feat_name);
        
        % Realizar clasificación
        [accuracies, precisions, recalls, specificities, f1_scores, ...
         train_times, predict_times,accuracy_knn,accuracy_svm,accuracy_lda,accuracy_rf,accuracy_mlp,accuracy_lstm] = clasifica_wrapper(balanced_features.(feat_name));
        
        % Almacenar resultados
        for clf_idx = 1:length(classifiers)
            clf_name = classifiers{clf_idx};
            
            % Métricas básicas
            results.(['a_' clf_name '_' feat_name]) = accuracies(clf_idx);
            results.(['p_' clf_name '_' feat_name]) = precisions(clf_idx);
            results.(['r_' clf_name '_' feat_name]) = recalls(clf_idx);
            results.(['s_' clf_name '_' feat_name]) = specificities(clf_idx);
            results.(['f1_' clf_name '_' feat_name]) = f1_scores(clf_idx);
            
            % Tiempos
            results.(['tt_' clf_name '_' feat_name]) = train_times(clf_idx);
            results.(['pt_' clf_name '_' feat_name]) = predict_times(clf_idx);
        end
    end

    %PARA PRUEBA ESTADISTICA...hay que guardar en la estructura data_user
    %las accuracy de cada fold de cada sujeto. luego aplicar la prueba de
    %friedman
    results.acc_users=[accuracy_knn(:),accuracy_svm(:),accuracy_lda(:),accuracy_rf(:),accuracy_mlp(:),accuracy_lstm(:)];

    % Almacenar todos los resultados para el usuario
    data_users{i,1} = results;
end

%% Función para prueba estadística de Friedman + post-hoc de Nemenyi
% data_db3 es ahora 135×6
% Nombres de columnas (para referencia):
nombres = {'KNN','SVM','LDA','RF','MLP','LSTM'};

data_db3=zeros(sujetos*5,6); % (sujetos x folds, clasificadores)
count=0;
for i=1:sujetos
    data_db3(count+1:count+5,:)=data_users{i,1}.acc_users;
    count=count+5;
end
[p_friedman, tbl, stats] = friedman(data_db3, 1, 'off');
fprintf('\n\n\n=== NINAPRO DB3 ===\n');
fprintf('Friedman chi2 = %.4f\n', tbl{2,5});
fprintf('p-value       = %.6f\n', p_friedman);

if p_friedman < 0.05
    fprintf('\n→ Diferencia significativa (p < 0.05). Proceder con post-hoc.\n\n');
    p_nemenyi = f_nemenyi_test(data_db3);
    f_mostrar_nemenyi(nombres,p_nemenyi,data_db3);
else
    fprintf('→ No hay diferencia significativa entre clasificadores.\n\n');
end

%% funci[on para dibujar la matriz de comfusion promedio
f_dibuja_cm_lda(mtx_mean_cm_mav_lda,gestos,true)
f_dibuja_cm_lstm(mtx_mean_cm_mav_lstm,gestos,true) 

% Función wrapper para organizar los resultados de clasificación
function [accuracies, precisions, recalls, specificities, f1_scores, ...
          train_times, predict_times,accuracy_knn,accuracy_svm,accuracy_lda,accuracy_rf,accuracy_mlp,accuracy_lstm] = clasifica_wrapper(data)
    
    % Realizar clasificación
    [a_knn, a_svm, a_lda, a_rf, a_mlp, a_lstm, ...
    P_knn, R_knn, S_knn, F1_knn,... 
    P_svm, R_svm, S_svm, F1_svm, P_lda, R_lda, S_lda, F1_lda, ...
     P_rf, R_rf, S_rf, F1_rf, ...
     P_mlp, R_mlp, S_mlp, F1_mlp, P_lstm, R_lstm, S_lstm, F1_lstm, ...
     train_time, predict_time,accuracy_knn,accuracy_svm,accuracy_lda,accuracy_rf,accuracy_mlp,accuracy_lstm] = clasifica(data);
 
    % Organizar resultados
    accuracies = [a_knn, a_svm, a_lda, a_rf, a_mlp, a_lstm];
    
    precisions = [
        P_knn, P_svm, P_lda, P_rf, P_mlp, P_lstm %los NaN es porque no calculo precision para knn,Naive Bayes
    ];
    
    recalls = [
        R_knn, R_svm, R_lda, R_rf, R_mlp, R_lstm %los NaN es porque no calculo recall para knn,Naive Bayes
    ];
    
    specificities = [
        S_knn, S_svm, S_lda, S_rf, S_mlp, S_lstm %los NaN es porque no calculo specificities para knn,Naive Bayes
    ];
    
    f1_scores = [
        F1_knn, F1_svm, F1_lda, F1_rf, F1_mlp, F1_lstm %los NaN es porque no calculo F1-score para knn,Naive Bayes
    ];
    
    train_times = train_time;
    predict_times = predict_time;
end

for i=1:sujetos
    a_knn_RMS(i)=data_users{i,1}.a_knn_RMS; a_svm_RMS(i)=data_users{i,1}.a_svm_RMS; a_lda_RMS(i)=data_users{i,1}.a_lda_RMS;
    a_rf_RMS(i)=data_users{i,1}.a_rf_RMS; a_mlp_RMS(i)=data_users{i,1}.a_mlp_RMS;a_lstm_RMS(i)=data_users{i,1}.a_lstm_RMS;
    
    a_knn_MAV(i)=data_users{i,1}.a_knn_MAV; a_svm_MAV(i)=data_users{i,1}.a_svm_MAV; a_lda_MAV(i)=data_users{i,1}.a_lda_MAV;
    a_rf_MAV(i)=data_users{i,1}.a_rf_MAV; a_mlp_MAV(i)=data_users{i,1}.a_mlp_MAV; a_lstm_MAV(i)=data_users{i,1}.a_lstm_MAV;
    
    a_knn_VAR(i)=data_users{i,1}.a_knn_VAR; a_svm_VAR(i)=data_users{i,1}.a_svm_VAR; a_lda_VAR(i)=data_users{i,1}.a_lda_VAR;
    a_rf_VAR(i)=data_users{i,1}.a_rf_VAR; a_mlp_VAR(i)=data_users{i,1}.a_mlp_VAR; a_lstm_VAR(i)=data_users{i,1}.a_lstm_VAR;
    
    a_knn_WL(i)=data_users{i,1}.a_knn_WL; a_svm_WL(i)=data_users{i,1}.a_svm_WL; a_lda_WL(i)=data_users{i,1}.a_lda_WL;
    a_rf_WL(i)=data_users{i,1}.a_rf_WL; a_mlp_WL(i)=data_users{i,1}.a_mlp_WL; a_lstm_WL(i)=data_users{i,1}.a_lstm_WL;

    a_knn_HIST(i)=data_users{i,1}.a_knn_HIST; a_svm_HIST(i)=data_users{i,1}.a_svm_HIST; a_lda_HIST(i)=data_users{i,1}.a_lda_HIST;
    a_rf_HIST(i)=data_users{i,1}.a_rf_HIST; a_mlp_HIST(i)=data_users{i,1}.a_mlp_HIST; a_lstm_HIST(i)=data_users{i,1}.a_lstm_HIST;
    
    a_knn_CC(i)=data_users{i,1}.a_knn_CC; a_svm_CC(i)=data_users{i,1}.a_svm_CC; a_lda_CC(i)=data_users{i,1}.a_lda_CC;
    a_rf_CC(i)=data_users{i,1}.a_rf_CC; a_mlp_CC(i)=data_users{i,1}.a_mlp_CC;a_lstm_CC(i)=data_users{i,1}.a_lstm_CC;

    a_knn_WAV(i)=data_users{i,1}.a_knn_WAV; a_svm_WAV(i)=data_users{i,1}.a_svm_WAV; a_lda_WAV(i)=data_users{i,1}.a_lda_WAV;
    a_rf_WAV(i)=data_users{i,1}.a_rf_WAV; a_mlp_WAV(i)=data_users{i,1}.a_mlp_WAV;a_lstm_WAV(i)=data_users{i,1}.a_lstm_WAV;
end

fprintf('\n\nBDD NINAPRO DB3');
fprintf('\nFECHA: %s\n', (datetime('now')));
fprintf('\nRESULTADOS FINALES - ACCURACY PROMEDIO DE CLASIFICADORES PARA LOS %d USUARIOS:',sujetos);
fprintf('\nMedia RMS: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_RMS),std(a_knn_RMS),mean(a_svm_RMS),std(a_svm_RMS),mean(a_lda_RMS),std(a_lda_RMS),mean(a_rf_RMS),std(a_rf_RMS),mean(a_mlp_RMS),std(a_mlp_RMS),mean(a_lstm_RMS),std(a_lstm_RMS));
fprintf('\nMedia MAV: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_MAV),std(a_knn_MAV),mean(a_svm_MAV),std(a_svm_MAV),mean(a_lda_MAV),std(a_lda_MAV),mean(a_rf_MAV),std(a_rf_MAV),mean(a_mlp_MAV),std(a_mlp_MAV),mean(a_lstm_MAV),std(a_lstm_MAV));
fprintf('\nMedia VAR: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_VAR),std(a_knn_VAR),mean(a_svm_VAR),std(a_svm_VAR),mean(a_lda_VAR),std(a_lda_VAR),mean(a_rf_VAR),std(a_rf_VAR),mean(a_mlp_VAR),std(a_mlp_VAR),mean(a_lstm_VAR),std(a_lstm_VAR));
fprintf('\nMedia WL: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_WL),std(a_knn_WL),mean(a_svm_WL),std(a_svm_WL),mean(a_lda_WL),std(a_lda_WL),mean(a_rf_WL),std(a_rf_WL),mean(a_mlp_WL),std(a_mlp_WL),mean(a_lstm_WL),std(a_lstm_WL));
fprintf('\nMedia HIST: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_HIST),std(a_knn_HIST),mean(a_svm_HIST),std(a_svm_HIST),mean(a_lda_HIST),std(a_lda_HIST),mean(a_rf_HIST),std(a_rf_HIST),mean(a_mlp_HIST),std(a_mlp_HIST),mean(a_lstm_HIST),std(a_lstm_HIST));
fprintf('\nMedia CC: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_CC),std(a_knn_CC),mean(a_svm_CC),std(a_svm_CC),mean(a_lda_CC),std(a_lda_CC),mean(a_rf_CC),std(a_rf_CC),mean(a_mlp_CC),std(a_mlp_CC),mean(a_lstm_CC),std(a_lstm_CC));
fprintf('\nMedia WAVELET: \tKNN: %0.2f±%0.2f \tSVM: %0.2f±%0.2f \tLDA: %0.2f±%0.2f \tRF: %0.2f±%0.2f \tMLP: %0.2f±%0.2f \tLSTM: %0.2f±%0.2f',mean(a_knn_WAV),std(a_knn_WAV),mean(a_svm_WAV),std(a_svm_WAV),mean(a_lda_WAV),std(a_lda_WAV),mean(a_rf_WAV),std(a_rf_WAV),mean(a_mlp_WAV),std(a_mlp_WAV),mean(a_lstm_WAV),std(a_lstm_WAV));

% %Graph Average Accuracy by Classifier
% f_boxplot(a_knn_RMS,a_svm_RMS,a_lda_RMS,a_rf_RMS,a_nb_RMS,a_mlp_RMS,a_lstm_RMS) %con RMS
% f_boxplot(a_knn_MAV,a_svm_MAV,a_lda_MAV,a_rf_MAV,a_nb_MAV,a_mlp_MAV,a_lstm_MAV) %con MAV
% f_boxplot(a_knn_VAR,a_svm_VAR,a_lda_VAR,a_rf_VAR,a_nb_VAR,a_mlp_VAR,a_lstm_VAR) %con VAR
% 
% %Grafica de barras
% f_barplot(a_knn_RMS,a_svm_RMS,a_lda_RMS,a_rf_RMS,a_nb_RMS,a_mlp_RMS,a_lstm_RMS) %con RMS
% f_barplot(a_knn_MAV,a_svm_MAV,a_lda_MAV,a_rf_MAV,a_nb_MAV,a_mlp_MAV,a_lstm_MAV) %con MAV
% f_barplot(a_knn_VAR,a_svm_VAR,a_lda_VAR,a_rf_VAR,a_nb_VAR,a_mlp_VAR,a_lstm_VAR) %con VAR
% 
% %Grafica de barras dos características
% f_barplot_2(a_knn_RMS,a_svm_RMS,a_lda_RMS,a_rf_RMS,a_mlp_RMS,a_lstm_RMS, ...
%     a_knn_MAV,a_svm_MAV,a_lda_MAV,a_rf_MAV,a_mlp_MAV,a_lstm_MAV) %con RMS y MAV

%Encero matrices de precision, recall, sensitivity, f1_score
%RMS
p_knn_RMS=zeros(1,sujetos);r_knn_RMS=zeros(1,sujetos);s_knn_RMS=zeros(1,sujetos);f1_knn_RMS=zeros(1,sujetos);
p_svm_RMS=zeros(1,sujetos);r_svm_RMS=zeros(1,sujetos);s_svm_RMS=zeros(1,sujetos);f1_svm_RMS=zeros(1,sujetos);
p_lda_RMS=zeros(1,sujetos);r_lda_RMS=zeros(1,sujetos);s_lda_RMS=zeros(1,sujetos);f1_lda_RMS=zeros(1,sujetos);
p_rf_RMS=zeros(1,sujetos);r_rf_RMS=zeros(1,sujetos);s_rf_RMS=zeros(1,sujetos);f1_rf_RMS=zeros(1,sujetos);
p_mlp_RMS=zeros(1,sujetos);r_mlp_RMS=zeros(1,sujetos);s_mlp_RMS=zeros(1,sujetos);f1_mlp_RMS=zeros(1,sujetos);
p_lstm_RMS=zeros(1,sujetos);r_lstm_RMS=zeros(1,sujetos);s_lstm_RMS=zeros(1,sujetos);f1_lstm_RMS=zeros(1,sujetos);
%MAV
p_knn_MAV=zeros(1,sujetos);r_knn_MAV=zeros(1,sujetos);s_knn_MAV=zeros(1,sujetos);f1_knn_MAV=zeros(1,sujetos);
p_svm_MAV=zeros(1,sujetos);r_svm_MAV=zeros(1,sujetos);s_svm_MAV=zeros(1,sujetos);f1_svm_MAV=zeros(1,sujetos);
p_lda_MAV=zeros(1,sujetos);r_lda_MAV=zeros(1,sujetos);s_lda_MAV=zeros(1,sujetos);f1_lda_MAV=zeros(1,sujetos);
p_rf_MAV=zeros(1,sujetos);r_rf_MAV=zeros(1,sujetos);s_rf_MAV=zeros(1,sujetos);f1_rf_MAV=zeros(1,sujetos);
p_mlp_MAV=zeros(1,sujetos);r_mlp_MAV=zeros(1,sujetos);s_mlp_MAV=zeros(1,sujetos);f1_mlp_MAV=zeros(1,sujetos);
p_lstm_MAV=zeros(1,sujetos);r_lstm_MAV=zeros(1,sujetos);s_lstm_MAV=zeros(1,sujetos);f1_lstm_MAV=zeros(1,sujetos);

for i=1:sujetos
    p_knn_RMS(i)=data_users{i,1}.p_knn_RMS; r_knn_RMS(i)=data_users{i,1}.r_knn_RMS; s_knn_RMS(i)=data_users{i,1}.s_knn_RMS; f1_knn_RMS(i)=data_users{i,1}.f1_knn_RMS;
    p_svm_RMS(i)=data_users{i,1}.p_svm_RMS; r_svm_RMS(i)=data_users{i,1}.r_svm_RMS; s_svm_RMS(i)=data_users{i,1}.s_svm_RMS; f1_svm_RMS(i)=data_users{i,1}.f1_svm_RMS;
    p_lda_RMS(i)=data_users{i,1}.p_lda_RMS; r_lda_RMS(i)=data_users{i,1}.r_lda_RMS; s_lda_RMS(i)=data_users{i,1}.s_lda_RMS; f1_lda_RMS(i)=data_users{i,1}.f1_lda_RMS;
    p_rf_RMS(i)=data_users{i,1}.p_rf_RMS; r_rf_RMS(i)=data_users{i,1}.r_rf_RMS; s_rf_RMS(i)=data_users{i,1}.s_rf_RMS; f1_rf_RMS(i)=data_users{i,1}.f1_rf_RMS;
    p_mlp_RMS(i)=data_users{i,1}.p_mlp_RMS; r_mlp_RMS(i)=data_users{i,1}.r_mlp_RMS; s_mlp_RMS(i)=data_users{i,1}.s_mlp_RMS; f1_mlp_RMS(i)=data_users{i,1}.f1_mlp_RMS;
    p_lstm_RMS(i)=data_users{i,1}.p_lstm_RMS; r_lstm_RMS(i)=data_users{i,1}.r_lstm_RMS; s_lstm_RMS(i)=data_users{i,1}.s_lstm_RMS; f1_lstm_RMS(i)=data_users{i,1}.f1_lstm_RMS;

    p_knn_MAV(i)=data_users{i,1}.p_knn_MAV; r_knn_MAV(i)=data_users{i,1}.r_knn_MAV; s_knn_MAV(i)=data_users{i,1}.s_knn_MAV; f1_knn_MAV(i)=data_users{i,1}.f1_knn_MAV;
    p_svm_MAV(i)=data_users{i,1}.p_svm_MAV; r_svm_MAV(i)=data_users{i,1}.r_svm_MAV; s_svm_MAV(i)=data_users{i,1}.s_svm_MAV; f1_svm_MAV(i)=data_users{i,1}.f1_svm_MAV;
    p_lda_MAV(i)=data_users{i,1}.p_lda_MAV; r_lda_MAV(i)=data_users{i,1}.r_lda_MAV; s_lda_MAV(i)=data_users{i,1}.s_lda_MAV; f1_lda_MAV(i)=data_users{i,1}.f1_lda_MAV;
    p_rf_MAV(i)=data_users{i,1}.p_rf_MAV; r_rf_MAV(i)=data_users{i,1}.r_rf_MAV; s_rf_MAV(i)=data_users{i,1}.s_rf_MAV; f1_rf_MAV(i)=data_users{i,1}.f1_rf_MAV;
    p_mlp_MAV(i)=data_users{i,1}.p_mlp_MAV; r_mlp_MAV(i)=data_users{i,1}.r_mlp_MAV; s_mlp_MAV(i)=data_users{i,1}.s_mlp_MAV; f1_mlp_MAV(i)=data_users{i,1}.f1_mlp_MAV;
    p_lstm_MAV(i)=data_users{i,1}.p_lstm_MAV; r_lstm_MAV(i)=data_users{i,1}.r_lstm_MAV; s_lstm_MAV(i)=data_users{i,1}.s_lstm_MAV; f1_lstm_MAV(i)=data_users{i,1}.f1_lstm_MAV;
end

fprintf('\n\nRESULTADOS FINALES METRICAS MATRIZ DE CONFUSION DE LOS %d USUARIOS:',sujetos);
fprintf('\nRMS-KNN (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_knn_RMS),std(p_knn_RMS),mean(r_knn_RMS),std(r_knn_RMS),mean(s_knn_RMS),std(s_knn_RMS),mean(f1_knn_RMS),std(f1_knn_RMS));
fprintf('\nRMS-SVM (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_svm_RMS),std(p_svm_RMS),mean(r_svm_RMS),std(r_svm_RMS),mean(s_svm_RMS),std(s_svm_RMS),mean(f1_svm_RMS),std(f1_svm_RMS));
fprintf('\nRMS-LDA (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_lda_RMS),std(p_lda_RMS),mean(r_lda_RMS),std(r_lda_RMS),mean(s_lda_RMS),std(s_lda_RMS),mean(f1_lda_RMS),std(f1_lda_RMS));
fprintf('\nRMS-RF (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_rf_RMS),std(p_rf_RMS),mean(r_rf_RMS),std(r_rf_RMS),mean(s_rf_RMS),std(s_rf_RMS),mean(f1_rf_RMS),std(f1_rf_RMS));
fprintf('\nRMS-MLP (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_mlp_RMS),std(p_mlp_RMS),mean(r_mlp_RMS),std(r_mlp_RMS),mean(s_mlp_RMS),std(s_mlp_RMS),mean(f1_mlp_RMS),std(f1_mlp_RMS));
fprintf('\nRMS-LSTM (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_lstm_RMS),std(p_lstm_RMS),mean(r_lstm_RMS),std(r_lstm_RMS),mean(s_lstm_RMS),std(s_lstm_RMS),mean(f1_lstm_RMS),std(f1_lstm_RMS));

fprintf('\n\nMAV-KNN (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_knn_MAV),std(p_knn_MAV),mean(r_knn_MAV),std(r_knn_MAV),mean(s_knn_MAV),std(s_knn_MAV),mean(f1_knn_MAV),std(f1_knn_MAV));
fprintf('\nMAV-SVM (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_svm_MAV),std(p_svm_MAV),mean(r_svm_MAV),std(r_svm_MAV),mean(s_svm_MAV),std(s_svm_MAV),mean(f1_svm_MAV),std(f1_svm_MAV));
fprintf('\nMAV-LDA (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_lda_MAV),std(p_lda_MAV),mean(r_lda_MAV),std(r_lda_MAV),mean(s_lda_MAV),std(s_lda_MAV),mean(f1_lda_MAV),std(f1_lda_MAV));
fprintf('\nMAV-RF (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_rf_MAV),std(p_rf_MAV),mean(r_rf_MAV),std(r_rf_MAV),mean(s_rf_MAV),std(s_rf_MAV),mean(f1_rf_MAV),std(f1_rf_MAV));
fprintf('\nMAV-MLP (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_mlp_MAV),std(p_mlp_MAV),mean(r_mlp_MAV),std(r_mlp_MAV),mean(s_mlp_MAV),std(s_mlp_MAV),mean(f1_mlp_MAV),std(f1_mlp_MAV));
fprintf('\nMAV-LSTM (media ± std: ==>> \tPrecision: %0.2f±%0.2f \tRecall: %0.2f±%0.2f \tSensitivity: %0.2f±%0.2f \tF1-score: %0.2f±%0.2f',mean(p_lstm_MAV),std(p_lstm_MAV),mean(r_lstm_MAV),std(r_lstm_MAV),mean(s_lstm_MAV),std(s_lstm_MAV),mean(f1_lstm_MAV),std(f1_lstm_MAV));

% f_boxplot_confusion(p_svm_RMS,r_svm_RMS,s_svm_RMS,f1_svm_RMS,...
%     p_lda_RMS,r_lda_RMS,s_lda_RMS,f1_lda_RMS,...
%     p_rf_RMS,r_rf_RMS,s_rf_RMS,f1_rf_RMS,...
%     p_mlp_RMS,r_mlp_RMS,s_mlp_RMS,f1_mlp_RMS,...
%     p_lstm_RMS,r_lstm_RMS,s_lstm_RMS,f1_lstm_RMS,...
%     p_svm_MAV,r_svm_MAV,s_svm_MAV,f1_svm_MAV,...
%     p_lda_MAV,r_lda_MAV,s_lda_MAV,f1_lda_MAV,...
%     p_rf_MAV,r_rf_MAV,s_rf_MAV,f1_rf_MAV,...
%     p_mlp_MAV,r_mlp_MAV,s_mlp_MAV,f1_mlp_MAV,...
%     p_lstm_MAV,r_lstm_MAV,s_lstm_MAV,f1_lstm_MAV...
%     )

tt_knn_RMS=zeros(1,sujetos);pt_knn_RMS=zeros(1,sujetos);
tt_svm_RMS=zeros(1,sujetos);pt_svm_RMS=zeros(1,sujetos);
tt_lda_RMS=zeros(1,sujetos);pt_lda_RMS=zeros(1,sujetos);
tt_rf_RMS=zeros(1,sujetos);pt_rf_RMS=zeros(1,sujetos);
tt_mlp_RMS=zeros(1,sujetos);pt_mlp_RMS=zeros(1,sujetos);
tt_lstm_RMS=zeros(1,sujetos);pt_lstm_RMS=zeros(1,sujetos);

tt_knn_MAV=zeros(1,sujetos);pt_knn_MAV=zeros(1,sujetos);
tt_svm_MAV=zeros(1,sujetos);pt_svm_MAV=zeros(1,sujetos);
tt_lda_MAV=zeros(1,sujetos);pt_lda_MAV=zeros(1,sujetos);
tt_rf_MAV=zeros(1,sujetos);pt_rf_MAV=zeros(1,sujetos);
tt_mlp_MAV=zeros(1,sujetos);pt_mlp_MAV=zeros(1,sujetos);
tt_lstm_MAV=zeros(1,sujetos);pt_lstm_MAV=zeros(1,sujetos);

for i=1:sujetos
    tt_knn_RMS(i)=data_users{i,1}.tt_knn_RMS; tt_svm_RMS(i)=data_users{i,1}.tt_svm_RMS;
    tt_lda_RMS(i)=data_users{i,1}.tt_lda_RMS; tt_rf_RMS(i)=data_users{i,1}.tt_rf_RMS;
    tt_mlp_RMS(i)=data_users{i,1}.tt_mlp_RMS; tt_lstm_RMS(i)=data_users{i,1}.tt_lstm_RMS;

    tt_knn_MAV(i)=data_users{i,1}.tt_knn_MAV; tt_svm_MAV(i)=data_users{i,1}.tt_svm_MAV;
    tt_lda_MAV(i)=data_users{i,1}.tt_lda_MAV; tt_rf_MAV(i)=data_users{i,1}.tt_rf_MAV;
    tt_mlp_MAV(i)=data_users{i,1}.tt_mlp_MAV; tt_lstm_MAV(i)=data_users{i,1}.tt_lstm_MAV;

    pt_knn_RMS(i)=data_users{i,1}.pt_knn_RMS; pt_svm_RMS(i)=data_users{i,1}.pt_svm_RMS;
    pt_lda_RMS(i)=data_users{i,1}.pt_lda_RMS; pt_rf_RMS(i)=data_users{i,1}.pt_rf_RMS;
    pt_mlp_RMS(i)=data_users{i,1}.pt_mlp_RMS; pt_lstm_RMS(i)=data_users{i,1}.pt_lstm_RMS;

    pt_knn_MAV(i)=data_users{i,1}.pt_knn_MAV; pt_svm_MAV(i)=data_users{i,1}.pt_svm_MAV;
    pt_lda_MAV(i)=data_users{i,1}.pt_lda_MAV; pt_rf_MAV(i)=data_users{i,1}.pt_rf_MAV;
    pt_mlp_MAV(i)=data_users{i,1}.pt_mlp_MAV; pt_lstm_MAV(i)=data_users{i,1}.pt_lstm_MAV;
end

fprintf('\n\nRESULTADOS FINALES METRICAS TIEMPOS DE ENTRENAMIENTO Y PREDICCIÓN -CON RMS- DE LOS %d USUARIOS:',sujetos);
fprintf('\nKNN-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_knn_RMS),std(tt_knn_RMS),mean(pt_knn_RMS),std(pt_knn_RMS));
fprintf('\nSVM-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_svm_RMS),std(tt_svm_RMS),mean(pt_svm_RMS),std(pt_svm_RMS));
fprintf('\nLDA-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_lda_RMS),std(tt_lda_RMS),mean(pt_lda_RMS),std(pt_lda_RMS));
fprintf('\nRF-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_rf_RMS),std(tt_rf_RMS),mean(pt_rf_RMS),std(pt_rf_RMS));
fprintf('\nMLP-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_mlp_RMS),std(tt_mlp_RMS),mean(pt_mlp_RMS),std(pt_mlp_RMS));
fprintf('\nLSTM-RMS--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_lstm_RMS),std(tt_lstm_RMS),mean(pt_lstm_RMS),std(pt_lstm_RMS));

fprintf('\n\nRESULTADOS FINALES METRICAS TIEMPOS DE ENTRENAMIENTO Y PREDICCIÓN -CON MAV- DE LOS %d USUARIOS: -- MAV',sujetos);
fprintf('\nKNN-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_knn_MAV),std(tt_knn_MAV),mean(pt_knn_MAV),std(pt_knn_MAV));
fprintf('\nSVM-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_svm_MAV),std(tt_svm_MAV),mean(pt_svm_MAV),std(pt_svm_MAV));
fprintf('\nLDA-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_lda_MAV),std(tt_lda_MAV),mean(pt_lda_MAV),std(pt_lda_MAV));
fprintf('\nRF-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_rf_MAV),std(tt_rf_MAV),mean(pt_rf_MAV),std(pt_rf_MAV));
fprintf('\nMLP-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_mlp_MAV),std(tt_mlp_MAV),mean(pt_mlp_MAV),std(pt_mlp_MAV));
fprintf('\nLSTM-MAV--Tiempos Promedios(media ± std ): Entrenamiento: %0.3f±%0.3f \t Predicción: %0.3f±%0.3f',mean(tt_lstm_MAV),std(tt_lstm_MAV),mean(pt_lstm_MAV),std(pt_lstm_MAV));

f_boxchart(a_knn_RMS, a_svm_RMS, a_lda_RMS, a_rf_RMS, a_mlp_RMS, a_lstm_RMS, ...
    a_knn_MAV, a_svm_MAV, a_lda_MAV, a_rf_MAV, a_mlp_MAV, a_lstm_MAV, ...
    a_knn_VAR, a_svm_VAR, a_lda_VAR, a_rf_VAR, a_mlp_VAR, a_lstm_VAR)

f_guarda_datos(a_knn_RMS, a_svm_RMS, a_lda_RMS, a_rf_RMS, a_mlp_RMS, a_lstm_RMS, ...
    a_knn_MAV, a_svm_MAV, a_lda_MAV, a_rf_MAV, a_mlp_MAV, a_lstm_MAV, ...
    a_knn_VAR, a_svm_VAR, a_lda_VAR, a_rf_VAR,a_mlp_VAR, a_lstm_VAR, ...    
    p_knn_RMS, r_knn_RMS, s_knn_RMS, f1_knn_RMS,......
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
    tt_knn_RMS, tt_svm_RMS, tt_lda_RMS, tt_rf_RMS, tt_mlp_RMS, tt_lstm_RMS, ...
    tt_knn_MAV, tt_svm_MAV, tt_lda_MAV, tt_rf_MAV, tt_mlp_MAV, tt_lstm_MAV, ...
    pt_knn_RMS, pt_svm_RMS, pt_lda_RMS, pt_rf_RMS, pt_mlp_RMS, pt_lstm_RMS, ...
    pt_knn_MAV, pt_svm_MAV, pt_lda_MAV, pt_rf_MAV, pt_mlp_MAV, pt_lstm_MAV,...
    mtx_mean_cm_mav_lda,mtx_mean_cm_mav_lstm,...
    data_db3)

fprintf('\n\nProcesamiento Finalizado\n');
delete(gcp)
%% --- FUNCIONES ------ FUNCIONES ------ FUNCIONES ------ FUNCIONES ---
%% --- FUNCIONES ------ FUNCIONES ------ FUNCIONES ------ FUNCIONES ---
%% --- FUNCIONES ------ FUNCIONES ------ FUNCIONES ------ FUNCIONES ---
%% Extracción de Características %% ===== RMS-MAV-VAR =====
function [signal_rms,signal_mav,signal_var]=process_signal(emg,gestos,repeticiones,electrodos)
    % emg ==>> [n_gestos, n_repeticiones, n_muestras, n_electrodos])
    cont=0;
    signal_rms=zeros(gestos * repeticiones,electrodos+1); signal_mav=zeros(gestos * repeticiones,electrodos+1); signal_var=zeros(gestos * repeticiones,electrodos+1);
    for i=1:gestos % recorro etiquetas
        % label_name=sprintf('label_%d',i); %selecciono etiqueta
        for j=1:repeticiones %recorro repeticiones (10 repeticiones)
            cont=cont+1;
            signal_rms(cont,1:end-1)=rms(squeeze((emg(i,j,:,:))));
            signal_rms(cont,end)=i;
            signal_mav(cont,1:end-1)=mean(squeeze((emg(i,j,:,:))));
            signal_mav(cont,end)=i;
            signal_var(cont,1:end-1)=var(squeeze((emg(i,j,:,:))));
            signal_var(cont,end)=i;
        end
    end
end
%% ===== Waveform lenght (WL) =====
function [mtx_WL]=extraer_signal_WL(emg,gestos,repeticiones,electrodos)
    mtx_WL=[]; fs=100; 
    current_row = 1; max_diff=52; salto=electrodos;  
    for i=1:gestos % recorro etiquetas 
        wl_ini=zeros(electrodos,max_diff); current_row1=1; salto1=electrodos; features2=zeros(electrodos,max_diff);
        for j=1:repeticiones %recorro repeticiones 
            for k=1:electrodos %recorro electrodos 
                segment(:,1) = emg(i,j,:,k); % Extraer segmento de señal
                aux=f_wl(segment,fs);
                aux = f_resizeVector(aux, max_diff);
                wl_ini(k,:)=aux;
            end
            features2(current_row1:current_row1+salto1-1,:)=wl_ini;
            current_row1 = current_row1 + salto1;
        end
        features2 = f_saca_promedios(features2,electrodos); %matriz promedio de características WL
        label(1:size(features2,1))=i; %etiqueto
        features1=[features2 label'];
        mtx_WL(current_row:current_row+salto-1,:)=features1;
        current_row = current_row + salto;
    end
end
%% ===== sEMG HISTOGRAM =====
function [mtx_HIST]=HIST_features(emg,gestos,repeticiones,electrodos)
    num_bins = 20; % Número de bins en el histograma
    salto=electrodos; current_row1=1;salto1=electrodos;%10;
    for i=1:gestos % recorro etiquetas
        features2=zeros(10,num_bins); current_row=1;
        for j=1:repeticiones %recorro repeticiones
            HIST_features = zeros(electrodos,num_bins); %repeticiones x num intervalos
            for k=1:electrodos %recorro electrodos
                segment(:,1) = emg(i,j,:,k); % Extraer segmento de señal      
                HIST_features(k, :) = f_hist(segment',num_bins);
            end
            features2(current_row:current_row+salto-1,:)=HIST_features;% mtx promedio por intervalo, por electrodo, por repeticion
            current_row=current_row+salto;
        end
        features2=f_saca_promedios(features2,electrodos);
        label(1:size(features2,1))=i; %ETIQUETO
        features3=[features2 label'];
        features4(current_row1:current_row1+salto1-1,:)=features3;
        current_row1 = current_row1 + salto1;
    end
    mtx_HIST=features4;
end
%% ===== COEFICIENTES CEPSTRALES =====
function [mtx_CC]=CC_features(emg,gestos,repeticiones,electrodos) % coeficientes cepstrales
    mtx_CC=[]; fs=100;
    for i=1:gestos % recorro etiquetas
        %% verificar todo
        features1=[];
        for j=1:repeticiones %recorro repeticiones 
            selected_coeffs=[];
            for k=1:electrodos %recorro electrodos 
                segment(:,1) = emg(i,j,:,k); % Extraer segmento de señal
                [vector_coeficientes]=f_cc(segment,fs);
                selected_coeffs(k,:) = vector_coeficientes; %matriz que guarda los CC en el orden de los electrodos (e1,e2,e3,e4,etc.)
            end
            features1= vertcat(features1,selected_coeffs);%guardo todas las matrices que genero con CC            
        end
        features1=f_saca_promedios(features1,electrodos);
        label(1:size(features1,1))=i; % ETIQUETO
        features2=[features1 label'];
        mtx_CC=vertcat(mtx_CC,features2); % guardo matrices normalizadas generadas con CC por electrodo por etiqueta
    end
end
%% ===== DISCRETE WAVELET TRANSFORM (DWT) =====
function [mtx_coefApprox]=DWT_features(emg,waveletName,numLevels,gestos,repeticiones,electrodos)
    max_diff=round(size(emg,3)/(numLevels*2));
    mtx_coefApprox=[];
    for i=1:gestos % recorro etiquetas
        features1=[];
        for j=1:repeticiones %recorro repeticiones 
            feat_coefApprox=[];
            for k=1:electrodos %recorro electrodos 
                segment(:,1) = emg(i,j,:,k); % Extraer segmento de señal
                [C, L] = wavedec(segment, numLevels, waveletName);% Aplicar la Transformada de Wavelet Discreta (DWT)
                % Extraer coeficientes de aproximación en cada nivel
                coefApprox = appcoef(C, L, waveletName); % Coeficientes de aproximación
                [vectorRecortado] = f_resizeVector(coefApprox', max_diff);
                coefApprox=vectorRecortado;
                feat_coefApprox=vertcat(feat_coefApprox,coefApprox); clear C L
            end
            features1=vertcat(features1,feat_coefApprox);% agrupo coefAprox por cada electrodo, por repetición
        end
        clear segment vectorRecortado
        features1=f_saca_promedios(features1,electrodos);
        label(1:size(features1,1))=i;
        features2=horzcat(features1,label');
        mtx_coefApprox=vertcat(mtx_coefApprox, features2);%agrupo coefApprox (x electrodo, x repetición, x etiqueta)
    end    
end
%% ===== CLASIFICADORES ===== CLASIFICADORES ===== CLASIFICADORES =====
%% ===== CLASIFICADORES ===== CLASIFICADORES ===== CLASIFICADORES =====
%% ===== CLASIFICADORES ===== CLASIFICADORES ===== CLASIFICADORES =====
function[acc_knn,acc_svm,acc_lda,acc_rf,acc_mlp,acc_lstm,...
        P_knn,R_knn,S_knn,F1_knn,...
        P_svm,R_svm,S_svm,F1_svm,P_lda,R_lda,S_lda,F1_lda,...
        P_rf,R_rf,S_rf,F1_rf,...
        P_mlp,R_mlp,S_mlp,F1_mlp,P_lstm,R_lstm,S_lstm,F1_lstm,...
        train_time,predict_time,...
        accuracy_knn,accuracy_svm,accuracy_lda,accuracy_rf,accuracy_mlp,accuracy_lstm]=clasifica(mtx_balance)
    
    % Iniciar pool paralelo si no existe
    if isempty(gcp('nocreate'))
        parpool('Threads');; % Usa todos los núcleos físicos
    end

    % caracteristicas1=mtx_balance;
    X = mtx_balance(:,1:end-1); % Características
    Y = mtx_balance(:,end);   % Clases

    % Prepartición de datos
    cv =  cvpartition(Y, 'KFold', 5, 'Stratify', true); % 8 folds estratificados    
    num_folds = cv.NumTestSets;

    accuracy_knn=zeros(1,num_folds);
    accuracy_svm=zeros(1,num_folds);
    accuracy_lda=zeros(1,num_folds);
    accuracy_rf=zeros(1,num_folds);
    accuracy_mlp=zeros(1,num_folds);
    accuracy_lstm=zeros(1,num_folds);

    trainTime_knn=zeros(1,num_folds); predictTime_knn=zeros(1,num_folds);
    trainTime_svm=zeros(1,num_folds); predictTime_svm=zeros(1,num_folds);
    trainTime_lda=zeros(1,num_folds); predictTime_lda=zeros(1,num_folds);
    trainTime_rf=zeros(1,num_folds); predictTime_rf=zeros(1,num_folds);
    trainTime_mlp=zeros(1,num_folds); predictTime_mlp=zeros(1,num_folds);
    trainTime_lstm=zeros(1,num_folds); predictTime_lstm=zeros(1,num_folds);

    mtx_metricas_knn=zeros(num_folds,4);
    mtx_metricas_svm=zeros(num_folds,4);
    mtx_metricas_lda=zeros(num_folds,4);
    mtx_metricas_rf=zeros(num_folds,4);
    mtx_metricas_mlp=zeros(num_folds,4);
    mtx_metricas_lstm=zeros(num_folds,4);   

%parfor i=1:num_folds
for i=1:num_folds

        % Usar partición precalculada
        trainIdx = training(cv, i);
        testIdx = test(cv, i);

        % Separar en entrenamiento y prueba
        Xtrain = X(trainIdx,:);
        Ytrain = Y(trainIdx);
        Xtest = X(testIdx,:);
        Ytest = Y(testIdx);

        % ---------------------------------------------------------------
        % CORRECCIÓN: Eliminar características con varianza cero en ENTRENAMIENTO
        % y también asegurar que no causen problemas en PRUEBA
        zero_var_mask_train = var(Xtrain, 0, 1) == 0;       
        Xtrain = Xtrain(:, ~zero_var_mask_train);
                
        % CORRECCIÓN ADICIONAL: Eliminar características constantes en PRUEBA
        zero_var_mask_test = var(Xtest, 0, 1) == 0;
        Xtest = Xtest(:, ~zero_var_mask_test);
        % ---------------------------------------------------------------

        %Normalización conjunto de entrenamiento y prueba
        [normaliza_features2,X_mean,X_std]=f_normaliza_Z_score(Xtrain);
        Xtrain=normaliza_features2;
        Xtest = (Xtest - X_mean) ./ X_std; % Aplica parámetros del entrenamiento

        % Reemplazar posibles NaN por 0 (debido a std=0)
        Xtrain(isnan(Xtrain)) = 0;
        Xtest(isnan(Xtest)) = 0;
        
        %% Entrenar clasificador k-NN ======================================
        tic
        modelo = fitcknn(Xtrain, Ytrain,'NumNeighbors', 5);
        trainTime_knn(i)=toc;
        % Predecir en el conjunto de prueba
        tic
        y_pred_knn = predict(modelo, Xtest);
        predictTime_knn(i)=toc;
        % Evaluar precisión
        accuracy_knn(i) = sum(y_pred_knn == Ytest) / length(Ytest);
        %----------------------------------------------------------------
        %[Precision,Recall,Specificity,F1]=f_metricas(Ytest, y_pred_svm);
        [P,R,S,F1] = f_metricas(Ytest, y_pred_knn);
        mtx_metricas_knn(i,:) = [P,R,S,F1]; 

        %% Entrenar clasificador SVM ======================================
        tic
        svm_model = fitcecoc(Xtrain, Ytrain);
        trainTime_svm(i)=toc;
        % Predecir en el conjunto de prueba
        tic
        y_pred_svm = predict(svm_model, Xtest);
        predictTime_svm(i)=toc;
        accuracy_svm(i) = sum(y_pred_svm == Ytest) / length(Ytest);
        %----------------------------------------------------------------
        %[Precision,Recall,Specificity,F1]=f_metricas(Ytest, y_pred_svm);
        [P,R,S,F1] = f_metricas(Ytest, y_pred_svm);
        mtx_metricas_svm(i,:) = [P,R,S,F1];  
   
        %% Entrenar clasificador LDA ======================================
        %lda_model=fitcdiscr(Xtrain, Ytrain, 'DiscrimType', 'linear');
        tic
        lda_model=fitcdiscr(Xtrain, Ytrain, 'DiscrimType', 'pseudoLinear');
        trainTime_lda(i)=toc;
        %lda_model=fitcdiscr(Xtrain, Ytrain, 'DiscrimType', 'diagLinear');
        % Predecir en el conjunto de prueba
        tic
        y_pred_lda = predict(lda_model, Xtest);
        predictTime_lda(i)=toc;
        accuracy_lda(i) = sum(y_pred_lda == Ytest) / length(Ytest);
        [P,R,S,F1] = f_metricas(Ytest, y_pred_lda);
        mtx_metricas_lda(i,:) = [P,R,S,F1];
        %% Entrenar clasificador Random Forest ======================================
        tic
        rf_model=fitcensemble(Xtrain, Ytrain, 'Method', 'Bag');
        trainTime_rf(i)=toc;
        tic
        y_pred_rf = predict(rf_model, Xtest);
        predictTime_rf(i)=toc;
        accuracy_rf(i) = sum(y_pred_rf == Ytest) / length(Ytest);
        
        [P,R,S,F1] = f_metricas(Ytest, y_pred_rf);
        mtx_metricas_rf(i,:) = [P,R,S,F1];
        
        %% Entrenar Red Neuronal MLP ======================================
        % Convertir etiquetas a categóricas
        YTrain_categorical = categorical(Ytrain);
        YTest_categorical = categorical(Ytest);
        % Crear y entrenar MLP
        tic
        mlp_model = fitcnet(Xtrain, YTrain_categorical, ...
            'LayerSizes', [64 32 64 32], ...
            'Activations', {'relu', 'relu', 'relu', 'relu'}, ...
            'Lambda', 0.001,...
            'IterationLimit', 45);%, ...  % Máximo de épocas
            %'Verbose', 1);              % Mostrar progreso
        trainTime_mlp(i)=toc;
        % Predecir y evaluar
        tic
        y_pred_mlp = predict(mlp_model, Xtest);
        predictTime_mlp(i)=toc;
        accuracy_mlp(i) = mean(y_pred_mlp == YTest_categorical);

        [P,R,S,F1]=f_metricas(YTest_categorical, y_pred_mlp);
        mtx_metricas_mlp(i,:) = [P,R,S,F1];

        %% Entrenar Red Neuronal LSTM ======================================
        XTrainLSTM = f_convert_to_lstm_format(Xtrain); %función para convertir la señal de prueba a formato LSTM
        XTestLSTM = f_convert_to_lstm_format(Xtest);
        % Convertir etiquetas a categóricas
        YTrain_categorical = categorical(Ytrain);
        YTest_categorical = categorical(Ytest);

        input_size = size(Xtrain, 2);  % = 10 (características) % 10 canales EMG
        num_classes = 17;      % 17 gestos  %%===>>>>> CORRECCIÓN PARA BDD NINAPRO DB3 <<<<<<===%%
        % Definir arquitectura LSTM
        layers = [ ...
            sequenceInputLayer(input_size)
            lstmLayer(100, 'OutputMode', 'sequence')  % Primera LSTM devuelve secuencia completa
            lstmLayer(50, 'OutputMode', 'last')
            fullyConnectedLayer(128)       % Nueva capa densa
            batchNormalizationLayer()      % Normalización para estabilizar el aprendizaje
            reluLayer()                    % Función de activación no lineal
            dropoutLayer(0.5)              % Regularización para evitar overfitting
            fullyConnectedLayer(num_classes)
            softmaxLayer
            classificationLayer()
            ];

        options = trainingOptions('adam', ...
            'MaxEpochs', 100, ...
            'MiniBatchSize', 8, ...
            'ExecutionEnvironment', 'cpu', ...
            'InitialLearnRate', 1e-2, ... % Tasa fija durante todo el entrenamiento
            'LearnRateSchedule', 'piecewise', ... % Reduce la tasa en momentos específicos
            'LearnRateDropPeriod', 20, ... % Cada 20 épocas, reduce la tasa
            'LearnRateDropFactor', 0.1, ... % Multiplica la tasa por 0.1
            'L2Regularization', 0.01, ... % Regularización para evitar sobreajuste
            'Shuffle', 'every-epoch', ...
            'Verbose', false);
        
        % Entrenar la red
        tic
        net = trainNetwork(XTrainLSTM, YTrain_categorical, layers, options);
        trainTime_lstm(i)=toc;
        % Evaluar en el fold de prueba
        tic
        y_pred_lstm = classify(net, XTestLSTM);
        predictTime_lstm(i)=toc;
        accuracy_lstm(i) = sum(y_pred_lstm == YTest_categorical)/numel(YTest_categorical);

        [P,R,S,F1]=f_metricas(YTest_categorical, y_pred_lstm);
        mtx_metricas_lstm(i,:) = [P,R,S,F1];

end

    acc_knn=mean(accuracy_knn) * 100;
    acc_svm=mean(accuracy_svm) * 100;
    acc_lda=mean(accuracy_lda) * 100;
    acc_rf=mean(accuracy_rf) * 100;
    acc_mlp=mean(accuracy_mlp) * 100;
    acc_lstm=mean(accuracy_lstm) * 100;

    metricas_knn=mean(mtx_metricas_knn); 
    P_knn=metricas_knn(1); %precision
    R_knn=metricas_knn(2); %recall
    S_knn=metricas_knn(3); %specificity
    F1_knn=metricas_knn(4); %F1-score
    metricas_svm=mean(mtx_metricas_svm); 
    P_svm=metricas_svm(1); %precision
    R_svm=metricas_svm(2); %recall
    S_svm=metricas_svm(3); %specificity
    F1_svm=metricas_svm(4); %F1-score
    metricas_lda=mean(mtx_metricas_lda); 
    P_lda=metricas_lda(1);
    R_lda=metricas_lda(2);
    S_lda=metricas_lda(3);
    F1_lda=metricas_lda(4);
    metricas_rf=mean(mtx_metricas_rf); 
    P_rf=metricas_rf(1);
    R_rf=metricas_rf(2);
    S_rf=metricas_rf(3);
    F1_rf=metricas_rf(4);
    metricas_mlp=mean(mtx_metricas_mlp); 
    P_mlp=metricas_mlp(1);
    R_mlp=metricas_mlp(2);
    S_mlp=metricas_mlp(3);
    F1_mlp=metricas_mlp(4);
    metricas_lstm=mean(mtx_metricas_lstm); 
    P_lstm=metricas_lstm(1);
    R_lstm=metricas_lstm(2);
    S_lstm=metricas_lstm(3);
    F1_lstm=metricas_lstm(4);

    train_time(1,1)=mean(trainTime_knn); predict_time(1,1)=mean(predictTime_knn);
    train_time(1,2)=mean(trainTime_svm); predict_time(1,2)=mean(predictTime_svm);
    train_time(1,3)=mean(trainTime_lda); predict_time(1,3)=mean(predictTime_lda);
    train_time(1,4)=mean(trainTime_rf); predict_time(1,4)=mean(predictTime_rf);
    train_time(1,6)=mean(trainTime_mlp); predict_time(1,6)=mean(predictTime_mlp);
    train_time(1,7)=mean(trainTime_lstm); predict_time(1,7)=mean(predictTime_lstm);
    
    fprintf('\nAccuracy Mean KNN: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_knn) * 100,P_knn,R_knn,S_knn,F1_knn);
    fprintf('\nAccuracy Mean SVM: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_svm) * 100,P_svm,R_svm,S_svm,F1_svm);
    fprintf('\nAccuracy Mean LDA: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_lda) * 100,P_lda,R_lda,S_lda,F1_lda);
    fprintf('\nAccuracy Mean RF: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_rf) * 100,P_rf,R_rf,S_rf,F1_rf);
    fprintf('\nAccuracy Mean MLP: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_mlp) * 100,P_mlp,R_mlp,S_mlp,F1_mlp);
    fprintf('\nAccuracy Mean LSTM: %.2f %% \tPrecision: %.2f \tRecall: %.2f \tSpecificity: %.2f \tF1: %.2f',mean(accuracy_lstm) * 100,P_lstm,R_lstm,S_lstm,F1_lstm);
end