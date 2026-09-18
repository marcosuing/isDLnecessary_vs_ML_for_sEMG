% =========================================================================
% SCRIPT: Cargar, organizar y exportar vectores de Accuracy a .CSV
% =========================================================================
%Directorio:
%C:\Users\Marco\Documents\MATLAB\EMG\Ninapro\lectura_accuracies

% 1. Listas de clasificadores y descriptores
clasificadores = {'KNN', 'SVM', 'LDA', 'RF', 'MLP', 'LSTM'};
descriptores   = {'RMS', 'MAV', 'VAR'};

%   ENTRADAS
%     acc_db1  : vector [27x1] accuracy (%) por sujeto - Ninapro DB1
%     acc_db3  : vector [11x1] accuracy (%) por sujeto - Ninapro DB3
%     acc_dba  : vector [18x1] accuracy (%) por sujeto - CapgMyo DB-a

n_clasif = length(clasificadores);
n_desc   = length(descriptores);
n_muestras = 27; % Dimensión de cada vector (1x27)

% Preasignación de espacio para la matriz (18 filas x 27 columnas)
matriz_accuracy = zeros(n_clasif * n_desc, n_muestras);
nombres_combinaciones = cell(1, n_clasif * n_desc);

fila = 1;

% 2. Recorrido para buscar y concatenar los vectores del Workspace
for i = 1:n_clasif
    for j = 1:n_desc
        
        % Nombre esperado (ejemplo: a_knn_MAV)
        nombre_var = sprintf('a_%s_%s', lower(clasificadores{i}), upper(descriptores{j}));
        
        % Etiqueta legible para la columna del CSV
        etiqueta = sprintf('%s_%s', clasificadores{i}, descriptores{j});
        nombres_combinaciones{fila} = etiqueta;
        
        % Verificar si la variable existe en el Workspace
        if exist(nombre_var, 'var')
            matriz_accuracy(fila, :) = eval(nombre_var);
        else
            % Alternativa por si está en mayúsculas (ejemplo: a_KNN_MAV)
            nombre_var_alt = sprintf('a_%s_%s', clasificadores{i}, upper(descriptores{j}));
            if exist(nombre_var_alt, 'var')
                matriz_accuracy(fila, :) = eval(nombre_var_alt);
            else
                warning('No se encontró la variable "%s" ni "%s" en el Workspace.', nombre_var, nombre_var_alt);
            end
        end
        
        fila = fila + 1;
    end
end

% 3. Matriz Transpuesta (27 filas/muestras x 18 modelos/columnas)
matriz_datos = matriz_accuracy';

% Creación de la tabla con encabezados
tabla_accuracy = array2table(matriz_datos, 'VariableNames', nombres_combinaciones);

% =========================================================================
% 4. GUARDAR EN ARCHIVO .CSV
% =========================================================================

nombre_csv = 'resultados_accuracy_db1_11_sep_2026.csv';

% OPCIÓN A (Recomendada): Guarda la tabla con nombres de columna
writetable(tabla_accuracy, nombre_csv);

% OPCIÓN B (Opcional): Si solo deseas guardar la matriz numérica pura (sin encabezados)
% writematrix(matriz_datos, 'matriz_pura.csv'); % (Para MATLAB R2019a o superior)
% csvwrite('matriz_pura.csv', matriz_datos);     % (Para versiones antiguas de MATLAB)

fprintf('¡Matriz exportada exitosamente como "%s"!\n', nombre_csv);