%% Sacar promedios de matriz de CC / WL / HISTOGRAM, electrodo por electrodo
function [normal_x_electrodo]=f_saca_promedios(normaliza_features2,electrodos)
    idx_elec=0; mtx_mean=zeros(electrodos,size(normaliza_features2,2));
    for i=1:electrodos % recorro repeticiones.
        mtx=zeros(electrodos,size(normaliza_features2,2)); 
        for k=1:(size(normaliza_features2,1)/electrodos) %recorro filas de 10 en 10, recorro todas las repeticiones
            if k == 1
                mtx(k,:)=normaliza_features2(k+idx_elec,:); 
            else
                mtx(k,:)=normaliza_features2(electrodos*(k-1)+idx_elec+1,:);
            end
        end
        mtx_mean(i,:)=mean(mtx); %mtx electrodo - promedio (coeficientes ceptrales)
        idx_elec=idx_elec+1;
    end
    normal_x_electrodo=mtx_mean;
end
%la matriz "normal_x_electrodo" contiene el promedio de las 10 repeticiones por cada medicion de
%cada electrodo.