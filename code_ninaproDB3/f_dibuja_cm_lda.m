function [] = f_dibuja_cm_lda(mtx_mean_cm,num_clases, mostrar)
    % cm_promedio=mean(mtx_mean_cm,3);
    % cm_std=std(mtx_mean_cm,0,3);
    clases=1:num_clases;

    % Inicializar matrices para promedios y desviaciones de los porcentajes
    cm_promedio_pct = zeros(num_clases, num_clases);
    cm_std_pct = zeros(num_clases, num_clases);
    
    % Calcular para cada clase (fila) la normalización
    for i = 1:num_clases
        % Extraer las filas de la clase i en todos los folds
        filas_i = squeeze(mtx_mean_cm(i, :, :)); % matriz num_clases x num_folds
        % Suma de cada fold para la clase i (total de muestras reales de esa clase en ese fold)
        suma_filas = sum(filas_i, 1); % vector 1 x num_folds
        
        % Normalizar cada fold: cada celda dividida por la suma de su fila
        % (evitar división por cero)
        filas_norm = filas_i ./ suma_filas; % cada columna es un fold
        
        % Calcular media y desviación estándar a través de los folds (ignorando NaN si alguna suma es cero)
        cm_promedio_pct(i, :) = mean(filas_norm, 2, 'omitnan')' * 100;
        cm_std_pct(i, :) = std(filas_norm, 0, 2, 'omitnan')' * 100;
    end

    % Crear matriz de celdas con formato 'media ± std'
    cm_tabla = cell(num_clases, num_clases);
    for i = 1:num_clases
        for j = 1:num_clases
            media = cm_promedio_pct(i, j);
            desv = cm_std_pct(i, j);
            cm_tabla{i, j} = sprintf('%.2f ± %.2f', media, desv);
        end
    end

    %% opcion 1
    % % Mostrar la tabla si se solicita
    % if mostrar
    %     % Crear una figura con uitable
    %     f = figure('Name', 'Matriz de Confusión Promedio ± STD', ...
    %                'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);
    % 
    %     % Nombres de las columnas y filas (clases)
    %     colnames = cellstr(strcat('Mov ', num2str(clases')));
    %     rownames = colnames;
    % 
    %     % Crear la tabla
    %     t = uitable(f, 'Data', cm_tabla, ...
    %                 'ColumnName', colnames, ...
    %                 'RowName', rownames, ...
    %                 'Position', [20, 20, 1120, 360]);
    % 
    %     % Ajustar ancho de columnas
    %     t.ColumnWidth = num2cell(repmat(80, 1, num_clases));
    % end

    %% OPCION 2
    if mostrar
        % Crear figura con tamaño adecuado
        figure('Name', 'Matriz de Confusión Promedio', ...
               'NumberTitle', 'off', 'Position', [100, 100, 900, 700]);
        
        % --- Definir posiciones manuales (en coordenadas normalizadas) ---
        % [left bottom width height]
        pos_superior = [0.13, 0.35, 0.77, 0.6];   % Heatmap superior (más alto)
        pos_inferior = [0.13, 0.13, 0.77, 0.15];  % Heatmap inferior (más bajo, mismo ancho)
        
        % --- Heatmap superior: matriz de confusión ---
        axes('Position', pos_superior);
        h_sup = heatmap(clases, clases, cm_promedio_pct, ...
                        'Colormap', sky, ...
                        'ColorbarVisible', 'off', ...
                        'CellLabelFormat', '%.1f%%', ...
                        'FontSize', 10);
        h_sup.XLabel = 'Clase Predicha';
        h_sup.YLabel = 'Clase Real';
        h_sup.Title = sprintf('Matriz de Confusión Promedio (%%)\nLDA - MAV\t DB3');
        
        % --- Calcular TPR y FNR ---
        TPR = diag(cm_promedio_pct)';   % Fila con los aciertos (diagonal)
        FNR = 100 - TPR;                % Fila con errores
        
        % Matriz de 2 filas: TPR arriba, FNR abajo
        metricas = [TPR; FNR];
        
        % --- Heatmap inferior: métricas ---
        axes('Position', pos_inferior);
        h_inf = heatmap(clases, {'TPR (%)', 'FNR (%)'}, metricas, ...
                        'Colormap', sky, ...            % Mapa de colores 'sky'
                        'ColorbarVisible', 'off', ...    % Sin barra de color para no recargar
                        'CellLabelFormat', '%.1f', ...   % Un decimal
                        'FontSize', 10);
        h_inf.XLabel = '';      % No repetir etiqueta de columnas (ya está arriba)
        h_inf.YLabel = '';
        h_inf.Title = '';
        
        % Opcional: Ajustar el tamaño de la fuente de las etiquetas de columna si se desea
        % h_inf.XDisplayLabels = '';  % Descomentar para ocultar etiquetas de columna
    end
end

