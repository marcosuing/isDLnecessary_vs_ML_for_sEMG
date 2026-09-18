function [Precision,Recall,Specificity,F1]=f_metricas(Y_true, y_pred_svm)
    % Calculo matriz de confusion
    C = confusionmat(Y_true, y_pred_svm);
    % Inicio variables
    n_clases = size(C, 1); % Número de clases
    metricas = struct();

    % Calcular métricas para cada clase
    for i = 1:n_clases
        TP = C(i, i); %True Positive
        FP = sum(C(:, i)) - TP; %False Positive
        FN = sum(C(i, :)) - TP; %False Negative
        TN = sum(C(:)) - TP - FP - FN; %True Negative
        % Precisión/Presicion
        metricas(i).Precision = TP / (TP + FP + eps); % eps evita división por 0
        % Sensibilidad/Recall
        metricas(i).Recall = TP / (TP + FN + eps);
        % Especificidad
        metricas(i).Specificity = TN / (TN + FP + eps);
        % F1-Score
        metricas(i).F1 = 2 * (metricas(i).Precision * metricas(i).Recall) / ...
                          (metricas(i).Precision + metricas(i).Recall + eps);
    end
    % % Accuracy global
    % accuracy = sum(diag(C)) / sum(C(:));
    
    % Calcular promedios (para multiclase)
    if n_clases > 2
        metricas_avg.Precision = mean([metricas.Precision]);
        metricas_avg.Recall = mean([metricas.Recall]);
        metricas_avg.Specificity = mean([metricas.Specificity]);
        metricas_avg.F1 = mean([metricas.F1]);
    end

    % % Mostrar resultados
    % disp('Matriz de Confusión:');
    % disp(C);
    % fprintf('Accuracy Global: %.2f%%\n', accuracy * 100);
    % 
    % if n_clases == 2
    %     disp('Métricas para Clase 1 (Binary):');
    %     fprintf('Precision: %.2f%%\n', metricas(2).Precision * 100);
    %     fprintf('Recall: %.2f%%\n', metricas(2).Recall * 100);
    %     fprintf('Specificity: %.2f%%\n', metricas(1).Specificity * 100);
    %     fprintf('F1-Score: %.2f%%\n', metricas(2).F1 * 100);
    % else
    %     disp('Métricas Promedio (Multiclase):');
    %     fprintf('Precision: %.2f%%\n', metricas_avg.Precision * 100);
    %     fprintf('Recall: %.2f%%\n', metricas_avg.Recall * 100);
    %     fprintf('Specificity: %.2f%%\n', metricas_avg.Specificity * 100);
    %     fprintf('F1-Score: %.2f%%\n', metricas_avg.F1 * 100);
    % end
    Precision=metricas_avg.Precision * 100;
    Recall=metricas_avg.Recall * 100;
    Specificity=metricas_avg.Specificity * 100;
    F1=metricas_avg.F1 * 100;
end