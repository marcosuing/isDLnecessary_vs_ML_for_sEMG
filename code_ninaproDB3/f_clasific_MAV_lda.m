function [cm_promedio]=f_clasific_MAV_lda(mtx_balance)
     % Extraer características (todas las columnas excepto la última) y etiquetas (última columna)
    features = mtx_balance(:, 1:end-1);
    labels = mtx_balance(:, end);

    % Eliminar características con varianza cero
    varianza = var(features);
    features = features(:, varianza > 0);

    % Obtener las clases únicas (para mantener el orden en las matrices de confusión)
    clases = unique(labels);
    num_clases = length(clases);    
    % Crear partición estratificada con 5 folds
    cv = cvpartition(labels, 'KFold', 5, 'Stratify', true);  
    num_folds = cv.NumTestSets;
    % Inicializar matriz acumulada de confusión (tamaño num_clases x num_clases)
    cm_folds = zeros(num_clases, num_clases,num_folds);   
    % Bucle sobre cada fold
    for i = 1:num_folds
        % Obtener índices de entrenamiento y prueba para este fold
        trainIdx = cv.training(i);
        testIdx = cv.test(i);        
        % Datos de entrenamiento y prueba
        X_train = features(trainIdx, :);
        y_train = labels(trainIdx);
        X_test = features(testIdx, :);
        y_test = labels(testIdx);

        % Normalizar con z-score (usando parámetros del entrenamiento)
        mu = mean(X_train);
        sigma = std(X_train);
        X_train = (X_train - mu) ./ sigma;
        X_test = (X_test - mu) ./ sigma;

        % Entrenar modelo LDA
        %ldaModel = fitcdiscr(X_train, y_train);
        ldaModel=fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear');
        % Predecir sobre el conjunto de prueba
        y_pred = predict(ldaModel, X_test);      
        % Calcular matriz de confusión para este fold, asegurando el orden de clases
        cm_fold = confusionmat(y_test, y_pred, 'Order', clases);       
        % Almacenar la matriz en el arreglo 3D
        cm_folds(:, :, i) = cm_fold;
    end
    % Calcular la matriz promedio a lo largo de los folds
    cm_promedio = mean(cm_folds, 3);
end