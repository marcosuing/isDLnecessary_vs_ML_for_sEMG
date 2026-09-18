function[resultados]=f_diferencias_rangos(rng_etiquetas,gestos)
    resultados = rng_etiquetas;
    for j=1:gestos+1
        nombre_campo = sprintf('label_%d', j-1);
        for k=1:size(resultados.(nombre_campo),1)
            aux=resultados.(nombre_campo)(k,2);
            aux1=resultados.(nombre_campo)(k,1);
            resultados.(nombre_campo)(k,3)=aux-aux1;
        end
    end
end