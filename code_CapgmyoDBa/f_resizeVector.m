function[vectorRecortado] = f_resizeVector(vector, longitudDeseada)
    if length(vector) > longitudDeseada
        % Si el vector es más largo de lo deseado, recórtalo
        vectorRecortado = vector(1:longitudDeseada);
    elseif length(vector) < longitudDeseada
        % Si el vector es más corto, puedes decidir si deseas llenar con ceros u otro valor.
        % En este ejemplo, se llenará con ceros.
        vectorRecortado = [vector, zeros(1, longitudDeseada - length(vector))];
    else
        % El vector ya tiene la longitud deseada, no se hace ningún cambio.
        vectorRecortado = vector;
    end
end