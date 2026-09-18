function [] = f_boxplot_confusion(p_svm_RMS,r_svm_RMS,s_svm_RMS,f1_svm_RMS,...
    p_lda_RMS,r_lda_RMS,s_lda_RMS,f1_lda_RMS,...
    p_rf_RMS,r_rf_RMS,s_rf_RMS,f1_rf_RMS,...
    p_mlp_RMS,r_mlp_RMS,s_mlp_RMS,f1_mlp_RMS,...
    p_lstm_RMS,r_lstm_RMS,s_lstm_RMS,f1_lstm_RMS,...
    p_svm_MAV,r_svm_MAV,s_svm_MAV,f1_svm_MAV,...
    p_lda_MAV,r_lda_MAV,s_lda_MAV,f1_lda_MAV,...
    p_rf_MAV,r_rf_MAV,s_rf_MAV,f1_rf_MAV,...
    p_mlp_MAV,r_mlp_MAV,s_mlp_MAV,f1_mlp_MAV,...
    p_lstm_MAV,r_lstm_MAV,s_lstm_MAV,f1_lstm_MAV...
    )

    figure
    subplot(2,4,1)
    boxplot([p_svm_RMS',p_lda_RMS',p_rf_RMS',p_mlp_RMS',p_lstm_RMS'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Precision - RMS')
    %add_median_labels(gca);

    subplot(2,4,2)
    boxplot([r_svm_RMS',r_lda_RMS',r_rf_RMS',r_mlp_RMS',r_lstm_RMS'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Recall - RMS')
    
    subplot(2,4,3)
    boxplot([s_svm_RMS',s_lda_RMS',s_rf_RMS',s_mlp_RMS',s_lstm_RMS'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Sensitivity - RMS')

    subplot(2,4,4)
    boxplot([f1_svm_RMS',f1_lda_RMS',f1_rf_RMS',f1_mlp_RMS',f1_lstm_RMS'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('F1-score - RMS')
    
    subplot(2,4,5)
    boxplot([p_svm_MAV',p_lda_MAV',p_rf_MAV',p_mlp_MAV',p_lstm_MAV'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Precision - MAV')

    subplot(2,4,6)
    boxplot([r_svm_MAV',r_lda_MAV',r_rf_MAV',r_mlp_MAV',r_lstm_MAV'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Recall - MAV')
    
    subplot(2,4,7)
    boxplot([s_svm_MAV',s_lda_MAV',s_rf_MAV',s_mlp_MAV',s_lstm_MAV'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('Sensitivity - MAV')

    subplot(2,4,8)
    boxplot([f1_svm_MAV',f1_lda_MAV',f1_rf_MAV',f1_mlp_MAV',f1_lstm_MAV'],'Notch','on','Labels',{'SVM','LDA','RF','MLP','LSTM'},'Whisker',1)
    title('F1-score - MAV')
    
    figure
    subplot(2,2,1)
    boxplot([p_svm_RMS',p_svm_MAV',p_lda_RMS',p_lda_MAV',p_rf_RMS',p_rf_MAV',p_mlp_RMS',p_mlp_MAV',p_lstm_RMS',p_lstm_MAV'],'Notch','on','Labels',{'RMS-SVM','MAV-SVM','RMS-LDA','MAV-LDA','RMS-RF','MAV-RF','RMS-MLP','MAV-MLP','RMS-LSTM','MAV-LSTM'},'Whisker',1)
    title('Precision')
    %add_median_labels(gca);

    subplot(2,2,2)
    boxplot([r_svm_RMS',r_svm_MAV',r_lda_RMS',r_lda_MAV',r_rf_RMS',r_rf_MAV',r_mlp_RMS',r_mlp_MAV',r_lstm_RMS',r_lstm_MAV'],'Notch','on','Labels',{'RMS-SVM','MAV-SVM','RMS-LDA','MAV-LDA','RMS-RF','MAV-RF','RMS-MLP','MAV-MLP','RMS-LSTM','MAV-LSTM'},'Whisker',1)
    title('Recall')

    subplot(2,2,3)
    boxplot([s_svm_RMS',s_svm_MAV',s_lda_RMS',s_lda_MAV',s_rf_RMS',s_rf_MAV',s_mlp_RMS',s_mlp_MAV',s_lstm_RMS',s_lstm_MAV'],'Notch','on','Labels',{'RMS-SVM','MAV-SVM','RMS-LDA','MAV-LDA','RMS-RF','MAV-RF','RMS-MLP','MAV-MLP','RMS-LSTM','MAV-LSTM'},'Whisker',1)
    title('Sensitivity')

    subplot(2,2,4)
    boxplot([f1_svm_RMS',f1_svm_MAV',f1_lda_RMS',f1_lda_MAV',f1_rf_RMS',f1_rf_MAV',f1_mlp_RMS',f1_mlp_MAV',f1_lstm_RMS',f1_lstm_MAV'],'Notch','on','Labels',{'RMS-SVM','MAV-SVM','RMS-LDA','MAV-LDA','RMS-RF','MAV-RF','RMS-MLP','MAV-MLP','RMS-LSTM','MAV-LSTM'},'Whisker',1)
    title('F1-score')    
end

% Función auxiliar para agregar etiquetas de mediana
function add_median_labels(ax)
    % Encontrar los objetos de línea que representan las medianas
    lines = findobj(ax, 'Tag', 'Median');
    
    % Obtener las posiciones de las cajas
    boxes = findobj(ax, 'Tag', 'Box');
    
    % Para cada mediana
    for i = 1:length(lines)
        % Obtener las coordenadas de la línea de mediana
        xData = get(lines(i), 'XData');
        yData = get(lines(i), 'YData');
        
        % La posición x es el punto medio de la línea
        xPos = mean(xData);
        
        % El valor de la mediana es el valor y (constante)
        medianValue = yData(1);
        
        % Crear el texto con 2 decimales
        textStr = sprintf('%.2f', medianValue);
        
        % Agregar el texto en la posición adecuada
        text(xPos, yData(1), textStr,...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'bottom',...
            'FontSize', 8,...
            'Color', 'k',...
            'FontWeight', 'bold');
    end
end

% mtx(:,1)=p_svm_RMS';
    % mtx(:,2)=r_svm_RMS';
    % mtx(:,3)=s_svm_RMS';
    % mtx(:,4)=f1_svm_RMS';
    % mtx(:,5)=p_rf_RMS';
    % mtx(:,6)=r_rf_RMS';
    % mtx(:,7)=s_rf_RMS';
    % mtx(:,8)=f1_rf_RMS';
    % mtx(:,9)=p_mlp_RMS';
    % mtx(:,10)=r_mlp_RMS';
    % mtx(:,11)=s_mlp_RMS';
    % mtx(:,12)=f1_mlp_RMS';
    % mtx(:,13)=p_svm_VAR';
    % mtx(:,14)=r_svm_VAR';
    % mtx(:,15)=s_svm_VAR';
    % mtx(:,16)=f1_svm_VAR';
    % mtx(:,17)=p_rf_VAR';
    % mtx(:,18)=r_rf_VAR';
    % mtx(:,19)=s_rf_VAR';
    % mtx(:,20)=f1_rf_VAR';
    % mtx(:,21)=p_mlp_VAR';
    % mtx(:,22)=r_mlp_VAR';
    % mtx(:,23)=s_mlp_VAR';
    % mtx(:,24)=f1_mlp_VAR';