%% Normalización Z-score
function [normaliza_features2,X_mean,X_std]=f_normaliza_Z_score(features2)
    mtx_features=double(features2);
    X_mean = mean(mtx_features, 1);  % Media por columna
    X_std = std(mtx_features, 0, 1); % Desviación estándar por columna
    X_std(X_std == 0) = 1; % Evita división por cero
    mtx_features_norm = (mtx_features - X_mean) ./ X_std;
    normaliza_features2=mtx_features_norm;
end