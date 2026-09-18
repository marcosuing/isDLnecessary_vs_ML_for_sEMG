function [] = f_mostrar_nemenyi(nombres,p_nemenyi,data_db1)
    %%  Ejecutar el post-hoc y mostrar resultados
    % Mostrar como tabla legible
    fprintf('\n=== Post-hoc Nemenyi — NINAPRO DB1 ===\n');
    fprintf('%-6s',' ');
    for i = 1:length(nombres)
        fprintf('%-8s', nombres{i});
    end
    fprintf('\n');
    
    for i = 1:length(nombres)
        fprintf('%-6s', nombres{i});
        for j = 1:length(nombres)
            if i == j
                fprintf('%-8s', '---');
            elseif p_nemenyi(i,j) < 0.001
                fprintf('%-8s', '<0.001*');
            elseif p_nemenyi(i,j) < 0.05
                fprintf('%-8s', sprintf('%.3f*', p_nemenyi(i,j)));
            else
                fprintf('%-8s', sprintf('%.3f', p_nemenyi(i,j)));
            end
        end
        fprintf('\n');
    end
    fprintf('\n* = diferencia significativa (p < 0.05)\n\n');

    %% Rangos promedio
    % Calcular rangos promedio de Friedman
    ranks_matrix = zeros(size(data_db1));
    for i = 1:size(data_db1,1)
        ranks_matrix(i,:) = tiedrank(data_db1(i,:));
    end
    
    avg_ranks = mean(ranks_matrix, 1);
    
    fprintf('\n=== Rangos promedio — NINAPRO DB1 ===\n');
    for i = 1:length(nombres)
        fprintf('  %s: %.4f\n', nombres{i}, avg_ranks(i));
    end
    % Rango más bajo = mejor clasificador

end