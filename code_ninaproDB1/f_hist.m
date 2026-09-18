function[conteo]=f_hist(segment,num_bins)
    % 1. Determinar límites dinámicamente
    data_range = prctile(segment, [0.5, 99.5]);  % Usar percentiles para evitar outliers
    limite_inferior = max(0, data_range(1));      % Mínimo no menor a 0
    limite_superior = min(0.6, data_range(2));    % Máximo no mayor a 0.6
    % 2. Asegurar rango válido
    if limite_superior <= limite_inferior
        limite_inferior = 0;
        limite_superior = 0.6;
    end
    % Crear bordes de los intervalos
    bordes = linspace(limite_inferior, limite_superior, num_bins + 1);
    %Calcular histogramas para cada señal
    % Normalización de la Señal
    %segment = (segment - mean(segment)) / std(segment);    
    %Calculo HISTOGRAMA

    % 4. Calcular histograma con normalización
    conteo = histcounts(segment, bordes, 'Normalization', 'probability');
    % 5. Asegurar vector fila
    conteo = conteo(:)';
end
