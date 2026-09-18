function p_matrix = f_nemenyi_test(data)
% Nemenyi post-hoc test para prueba de Friedman
% INPUT:  data  - matriz N×k (N observaciones, k clasificadores)
% OUTPUT: p_matrix - matriz k×k de p-values entre pares

    [N, k] = size(data);
    
    % Calcular rangos por fila
    ranks = zeros(N, k);
    for i = 1:N
        ranks(i,:) = tiedrank(data(i,:));
    end
    
    % Rango promedio por clasificador
    avg_ranks = mean(ranks, 1);
    
    % Diferencia crítica y estadístico
    num_comparisons = k*(k-1)/2;
    p_matrix = ones(k, k);
    
    for i = 1:k
        for j = (i+1):k
            % Estadístico z de Nemenyi
            z = (avg_ranks(i) - avg_ranks(j)) / ...
                sqrt(k*(k+1) / (6*N));
            
            % p-value bilateral basado en distribución normal
            p_ij = 2 * (1 - normcdf(abs(z)));
            
            % Corrección de Bonferroni para comparaciones múltiples
            p_matrix(i,j) = min(p_ij * num_comparisons, 1);
            p_matrix(j,i) = p_matrix(i,j);
        end
    end
end