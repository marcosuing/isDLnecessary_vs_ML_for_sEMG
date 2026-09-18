function[data_resample]=f_resample_ninaprodb1(all_data,rng_etiquetasUser,min_diff,sujetos,gestos,repeticiones,electrodos,factor)
    % Paper: Electromyography data for non-invasive naturally-controlled robotic hand prostheses
    % link: https://www.nature.com/articles/sdata201453    
    
    % Preasignar matriz de resultados
    data_resample = zeros(sujetos, gestos, repeticiones, ceil(min_diff/factor), electrodos);
    %% Preprocesar la estructura rng_etiquetasUser para evitar variables de broadcast
    % Crear una matriz de índices compacta y convertir a tipo numérico
    indices = zeros(sujetos, gestos, repeticiones, 2);
    
    for suj = 1:sujetos
        campo_usuario = sprintf('U%d', suj);
        for mov = 1:gestos
            campo_etiqueta = sprintf('label_%d', mov);
            for rep = 1:repeticiones
                indices(suj, mov, rep, 1) = double(rng_etiquetasUser.(campo_usuario).(campo_etiqueta)(rep, 1));
                indices(suj, mov, rep, 2) = double(rng_etiquetasUser.(campo_usuario).(campo_etiqueta)(rep, 2));
            end
        end
    end

    % Iniciar pool paralelo si no existe
    % if isempty(gcp('nocreate'))
    %     parpool('Processes'); % Usa todos los núcleos físicos
    % end

    for suj = 1:sujetos
        % Obtener los datos específicos para este sujeto
        sujeto_indices = squeeze(indices(suj, :, :, :)); %sujeto-gesto-repeticion-indices
        sujeto_data = all_data{suj, 1}.emg;
        % Preasignar matriz temporal para este sujeto
        sujeto_resample = zeros(gestos, repeticiones, ceil(min_diff/factor), electrodos);

        for mov = 1: gestos
            for rep = 1:repeticiones
                indx_inicio = sujeto_indices(mov, rep, 1);
                indx_fin = sujeto_indices(mov, rep, 2);
                for elect=1:electrodos
                    % Extraer segmento de señal
                    segment = sujeto_data(indx_inicio:indx_fin, elect);                    
                    % Redimensionar y remuestrear
                    segment = f_resizeVector(segment', min_diff);
                    segment = resample(segment, 1, factor);
                    segment = f_acondicionar(segment);
                    sujeto_resample(mov, rep, :, elect) =segment';
                    
                    %código original
                    %sujeto_resample(mov, rep, :, elect) = resample(segment', 1, factor);
                end
            end
        end
        % Almacenar resultados
        data_resample(suj,:,:,:,:) = sujeto_resample;%sujeto-gesto-repeticion-muestras-electrodos
    end   
    fprintf('\nResampleo Finalizado\n');
end
%% versión original

% % Iniciar pool paralelo si no existe
%     if isempty(gcp('nocreate'))
%         parpool('Processes'); % Usa todos los núcleos físicos
%     end
% 
%     parfor suj = 1:sujetos
%         emg_resample=zeros(ceil(min_diff/factor),electrodos);
%         for mov = 1: gestos
%             for rep = 1:repeticiones
%                 indx_inicio=rng_etiquetasUser.(sprintf('U%d',suj)).(sprintf('label_%d',mov))(rep,1);
%                 indx_fin=rng_etiquetasUser.(sprintf('U%d',suj)).(sprintf('label_%d',mov))(rep,2);
%                 for elect=1:electrodos
%                     signal=all_data{suj,1}.emg(:,elect);
%                     % Extract the segment of the signal to be resampled
%                     segment = signal(indx_inicio:indx_fin);
%                     segment = f_resizeVector(segment',min_diff);
%                     % Resample the extracted segment
%                     emg_resample(:, elect) = resample(segment', 1, factor);
%                 end
%                 % Almacenar resultados
%                 data_resample(suj,mov,rep,:,:) = emg_resample;%sujeto-gesto-repeticion-muestras-electrodos
%             end
%         end
%     end