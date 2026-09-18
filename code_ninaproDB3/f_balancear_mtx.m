%% Función de Balanceo de Matrices
%la matriz de caracteristicas debe ser horizontal con las etiquetas en la
%ultima columna
function[matriz_balanceada]=f_balancear_mtx(features2,gestos)
    etiquetas=features2(:,end);
    conteo_etiquetas = histcounts(etiquetas, 1:gestos+1); % ADAPTACIÓN DE ETIQUETAS A BDD CAPGMYO
    % Mostrar el resultado
    % disp('Conteo de etiquetas:');
    % for i = 1:13
    %     fprintf('Etiqueta %d: %d veces\n', i-1, conteo_etiquetas(i));
    % end
    min_muestras=min(conteo_etiquetas);
    %fprintf('\nLa etiqueta minoritaria tiene %d muestras.', min_muestras);
    % Inicializar matriz balanceada
    matriz_balanceada = zeros(gestos * min_muestras, size(features2, 2));
    current_row = 1;
    % Para cada etiqueta, seleccionar 'min_muestras' filas aleatorias
    for etiqueta = 1:gestos
        idx = find(etiquetas == etiqueta); % Obtener índices de las filas con esta etiqueta
        % Mezclar los índices y seleccionar los primeros 'min_muestras'
        idx_aleatorio = idx(randperm(length(idx))); 
        idx_seleccionados = idx_aleatorio(1:min_muestras);    
        % Añadir filas seleccionadas a la matriz balanceada
        matriz_balanceada(current_row:current_row + min_muestras - 1, :) = features2(idx_seleccionados, :);
        current_row = current_row + min_muestras;
    end
    %fprintf('\nTamaño de la matriz balanceada: %d x %d', size(matriz_balanceada));
end