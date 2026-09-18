%% Encontrar rangos de etiquetas:
function [resultados]= f_rangos_etiquetas(stimulus)
    etiquetas = stimulus; % Tu vector real (101014x1)
    % 1. Encontrar puntos donde cambian las etiquetas
    cambios = find(diff(etiquetas) ~= 0); % Índices donde hay cambio
    cambios = [0; cambios; length(etiquetas)]; % Añadir inicio y final
    % 2. Calcular rangos [inicio, fin] para cada bloque
    rangos = [cambios(1:end-1) + 1, cambios(2:end)];
    % 3. Obtener las etiquetas correspondientes a cada bloque
    etiquetas_bloques = etiquetas(rangos(:,1));
    % 4. Almacenar rangos por etiqueta en una estructura
    resultados = struct();
    for i = 1:length(etiquetas_bloques)
        etiqueta_actual = etiquetas_bloques(i);
        nombre_campo = sprintf('label_%d', etiqueta_actual);
        
        if isfield(resultados, nombre_campo)
            resultados.(nombre_campo) = [resultados.(nombre_campo); rangos(i,:)];
        else
            resultados.(nombre_campo) = rangos(i,:);
        end
    end
    % % 5. Mostrar resultados organizados
    % for etiqueta = 0:12
    %     nombre_campo = sprintf('label_%d', etiqueta);
    %     if isfield(resultados, nombre_campo)
    %         fprintf('\nEtiqueta %d:', etiqueta);
    %         fprintf('\nBloques:\n');
    %         disp(resultados.(nombre_campo));
    %     else
    %         fprintf('\nEtiqueta %d: Sin bloques\n', etiqueta);
    %     end
    % end
end