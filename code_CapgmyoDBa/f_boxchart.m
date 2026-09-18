function [] = f_boxchart(a_knn_RMS, a_svm_RMS, a_lda_RMS, a_rf_RMS, a_mlp_RMS, a_lstm_RMS, ...
    a_knn_MAV, a_svm_MAV, a_lda_MAV, a_rf_MAV, a_mlp_MAV, a_lstm_MAV, ...
    a_knn_VAR, a_svm_VAR, a_lda_VAR, a_rf_VAR, a_mlp_VAR, a_lstm_VAR)

    % Preparar datos para Machine Learning
    datos_ML = {a_knn_RMS, a_knn_MAV, a_knn_VAR,...
                a_svm_RMS, a_svm_MAV, a_svm_VAR,...
                a_lda_RMS, a_lda_MAV, a_lda_VAR,...
                a_rf_RMS, a_rf_MAV, a_rf_VAR};
    
    etiquetas_ML = {};
    for i = 1:4
        etiquetas_ML = [etiquetas_ML, {'RMS', 'MAV', 'VAR'}];
    end
    
    clasificadores_ML = {'k-NN', 'SVM', 'LDA', 'RF'};
    posiciones_ML = [1,2,3, 5,6,7, 9,10,11, 13,14,15];

    % Preparar datos para Deep Learning
    datos_DL = {a_mlp_RMS, a_mlp_MAV, a_mlp_VAR,...
                a_lstm_RMS, a_lstm_MAV, a_lstm_VAR};
    
    etiquetas_DL = {'RMS', 'MAV', 'VAR', 'RMS', 'MAV', 'VAR'};
    clasificadores_DL = {'MLP', 'LSTM'};
    posiciones_DL = [1,2,3, 5,6,7];

    % Crear figura
    figure('Position', [100, 100, 1200, 500]);
    
    % Subplot 1: Machine Learning
    subplot(1, 2, 1);
    boxplot_grupos(datos_ML, etiquetas_ML, clasificadores_ML, posiciones_ML, 'Machine Learning Classifiers');
    
    % Subplot 2: Deep Learning
    subplot(1, 2, 2);
    boxplot_grupos(datos_DL, etiquetas_DL, clasificadores_DL, posiciones_DL, 'Deep Learning Classifiers');
end

function boxplot_grupos(datos, etiquetas, clasificadores, posiciones, titulo)
    % Convertir datos a vector
    datos_vector = [];
    grupos = [];
    
    for i = 1:length(datos)
        datos_vector = [datos_vector, datos{i}];
        grupos = [grupos, i * ones(1, length(datos{i}))];
    end
    
    % Crear boxplot
    boxplot(datos_vector, grupos, 'positions', posiciones, 'labels', etiquetas);
    
    % Colorear las cajas
    h = findobj(gca, 'Tag', 'Box');
    for j = 1:length(h)
        % Determinar el color según la característica
        if mod(j, 3) == 1 % RMS
            color = [1, 1, 0.6]; % Amarillo claro
        elseif mod(j, 3) == 2 % MAV
            color = [0.6, 0.6, 1]; % Azul claro
        else % VAR
            color = [1, 0.6, 0.6]; % Rojo claro
        end
        
        patch(get(h(j), 'XData'), get(h(j), 'YData'), color, 'FaceAlpha', 0.7);
    end
    
    % Personalizar ejes y título
    title(titulo);
    ylabel('Accuracy (%)');
    ylim([5 90]);
    grid on;
    set(gca, 'GridAlpha', 0.2);
    
    % Añadir etiquetas de clasificadores
    xlim([0 max(posiciones)+1]);
    for i = 1:length(clasificadores)
        text(mean(posiciones((i-1)*3+1:i*3)), 8, clasificadores{i},...
            'HorizontalAlignment', 'center','FontWeight','bold');
    end
    
    % Rotar etiquetas del eje X
    set(gca, 'XTickLabelRotation', 45);
end