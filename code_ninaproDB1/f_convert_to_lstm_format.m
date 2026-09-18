function [X_cell] = f_convert_to_lstm_format(X)
    % Convertir matriz [muestras x features] a celdas LSTM
    X_perm = permute(X, [2, 1]);  % [features, muestras]
    X_cell = mat2cell(...
        X_perm, ...
        size(X_perm, 1), ...    % Mantener todas las features
        ones(1, size(X_perm, 2))'); % Una columna por muestra
end