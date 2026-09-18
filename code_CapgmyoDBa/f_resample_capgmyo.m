function[data_resample]=f_resample_capgmyo(sujetos,gestos,repeticiones,electrodos,factor)
    %% Paper: Gesture Recognition by instantaneous surface EMG images.
    %  link: https://www.nature.com/articles/srep36571#Tab1
    % función para remuestrear la señal original de 1000 Hz a 100 Hz y
    % luego sacar promedio de las 10 repeticiones por cada movimiento por
    % cada sujeto.
    %% Configuración inicial
    baseDir = 'C:\Users\Marco\Documents\MATLAB\EMG\CapgMyo\db_a';
    % sujetos = 18;
    % gestos = 8;
    % repeticiones = 10;
    % f_original = 1000;
    % f_nueva = 100;
    % factor = f_original / f_nueva;  % Factor de resampleo
    
    % Precalcular tamaño de muestras resampleadas
    archivo_ejemplo = fullfile(baseDir, 'dba-s1', '001-001-001.mat');
    datos_ej = load(archivo_ejemplo);
    muestras_original = size(datos_ej.data, 1);
    muestras_resample = ceil(muestras_original / factor);
    %num_electrodos = size(datos_ej.data, 2);
    num_electrodos = electrodos;
    
    % Preasignar matriz de resultados
    data_resample = zeros(sujetos, gestos, repeticiones, muestras_resample, num_electrodos);
    
    % Iniciar pool paralelo si no existe
    % if isempty(gcp('nocreate'))
    %     parpool('Processes'); % Usa todos los núcleos físicos
    % end

    %parfor i = 1:sujetos
    for i = 1:sujetos
        folderName = sprintf('dba-s%d', i);
        folderPath = fullfile(baseDir, folderName);
        
        for j = 1:gestos
            for k = 1:repeticiones
                archivo = sprintf('%03d-%03d-%03d.mat', i, j, k);
                archivo_path = fullfile(folderPath, archivo);
                datos = load(archivo_path);
                emg_data = datos.data;
                
                % Preasignar matriz para resultados
                emg_resample = zeros(muestras_resample, num_electrodos);
                
                for l = 1:num_electrodos
                    % emg_limpia = detrend(emg_data(:, i)); %elimina componentes de DC antes del remuestreo
                    % emg_resample(:,l) =resample(emg_limpia, 1, factor);

                    segment=resample(emg_data(:,l), 1, factor);
                    segment=f_resizeVector(segment',muestras_resample);
                    segment=f_acondicionar(segment);
                    emg_resample(:,l) = segment';
                    
                    %código original
                    %emg_resample(:,l) =resample(emg_data(:,l), 1, factor);

                    % % Decimación con filtro FIR incorporado (optimizado)
                    % emg_resample(:,l) = decimate(emg_data(:,l), factor, 'FIR');
                end
                
                % Almacenar resultados
                data_resample(i,j,k,:,:) = emg_resample;%sujeto-gesto-repeticion-muestras-electrodos
            end
        end
    end
    fprintf('\nResampleo Finalizado\n');
end
%% CODIGO ORIGINAL
    % baseDir = 'C:\Users\Marco\Documents\MATLAB\EMG\CapgMyo\db_a';
    % % ================= RESAMPLEO a 100 Hz. ===========================
    % sujetos=18;
    % gestos=8;
    % repeticiones=10;
    % f_original=1000; 
    % f_nueva=100;
    % data_resample=zeros(0,0,0,0,0);
    % for i=1:sujetos
    %     folderName = sprintf('dba-s%d', i);
    %     folderPath = fullfile(baseDir, folderName);
    %     for j=1:gestos
    %         for k=1:repeticiones
    %             archivo = sprintf('%03d-%03d-%03d.mat',i,j,k);
    %             archivo_path = fullfile(folderPath, archivo);
    %             datos = load(archivo_path);
    %             emg_data=datos.data;
    %             % codigo para REMUESTREAR las señales electrodo por electrodo.
    %             for l=1:size(emg_data,2)
    %                 emg_resample=f_resample(emg_data(:,l),f_original,f_nueva);
    %                 data_resample(i,j,k,:,l)=emg_resample; %sujeto-gesto-repeticion-muestra-electrodo
    %             end
    %         end
    %     end
    % end
    % fprintf('\nResampleo Finalizado\n');
    % % ================= PROMEDIAR SEÑALES ===========================
    % % % Calcular el promedio a lo largo de la dimensión de repeticiones (dim3)
    % % promedio_gestos = mean(data_resample, 3);
    % % % Redimensionar para eliminar la dimensión de repeticiones
    % % promedio_gestos = reshape(promedio_gestos, [size(data_resample,1), size(data_resample,2), size(data_resample,4), size(data_resample,5)]);