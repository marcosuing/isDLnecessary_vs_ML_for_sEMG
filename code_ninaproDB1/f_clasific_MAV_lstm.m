function [cm_promedio]=f_clasific_MAV_lstm(mtx_balance)
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

        XTrainLSTM = f_convert_to_lstm_format(X_train);
        XTestLSTM = f_convert_to_lstm_format(X_test);
        % Convertir etiquetas a categóricas
        YTrain_categorical = categorical(y_train);

        input_size = size(X_train, 2);  % = 10 (características) % 10 canales EMG
        num_classes = 12;      % 12 gestos

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
            classificationLayer
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
        net = trainNetwork(XTrainLSTM, YTrain_categorical, layers, options);        
        % Evaluar en el fold de prueba
        % Predecir
        y_pred = classify(net, XTestLSTM);
        y_pred = double(y_pred);  % <--- CONVERSIÓN CLAVE

        % Calcular matriz de confusión para este fold, asegurando el orden de clases
        cm_fold = confusionmat(y_test, y_pred, 'Order', clases);    
        %cm_fold = confusionmat(YTest_categorical, y_pred, 'Order', clases);
        % Almacenar la matriz en el arreglo 3D
        cm_folds(:, :, i) = cm_fold;
    end
    % Calcular la matriz promedio a lo largo de los folds
    cm_promedio = mean(cm_folds, 3);
end