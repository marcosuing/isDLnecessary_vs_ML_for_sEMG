%% Maxima Diferencia entre Rangos de Etiquetas
function[min_diff]=f_minima_diferencia(rng_etiquetasUser,sujetos,gestos)
    min_dif=zeros(1,sujetos);
    suj=randi(sujetos);
    mov=randi(gestos);
    for i=1:sujetos
        label_name=sprintf('label_%d',mov);
        min_dif(i)=min(rng_etiquetasUser.(sprintf('U%d',suj)).(label_name)(:,3));
    end
    min_diff=fix(mean(min_dif));% redondeo al entero menor. 
end