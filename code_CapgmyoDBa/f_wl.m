function[wl_normalizado]=f_wl(segment,fs)
    %% Calcular Waveform Length (WL) para toda la señal
    %t = 0:1/fs:5-1/fs;
    N=size(segment,1);
    %wl_total = sum(abs(diff(segment)));
    
    %% Calcular WL en ventanas deslizantes (para análisis temporal)
    ventana_t = 0.2;                % Tamaño de ventana en segundos
    muestras_ventana = ventana_t * fs;  % Muestras por ventana (20 muestras)
    paso = round(muestras_ventana/2);   % Paso de deslizamiento (10 muestras)
    
    % Inicializar variables
    num_ventanas = floor((N - muestras_ventana)/paso) + 1;
    wl_ventanas = zeros(1, num_ventanas);
    %tiempos_wl = zeros(1, num_ventanas);
    %N=(1:size(segment,1));
    % Procesar cada ventana
    for i = 1:num_ventanas
        idx_inicio = (i-1)*paso + 1;
        idx_fin = idx_inicio + muestras_ventana - 1;
        ventana = segment(idx_inicio:idx_fin);
        wl_ventanas(i) = sum(abs(diff(ventana)));
        %tiempos_wl(i) = N(idx_inicio + round(muestras_ventana/2)); % Tiempo en centro de ventana
    end
    wl_normalizado = wl_ventanas; %/ max(wl_ventanas);
%% Visualización
% figure('Color', 'white', 'Name', 'Análisis Waveform Length')
% subplot(2,1,1)
% plot(N, segment', 'b', 'LineWidth', 1)
% title('Señal EMG sintética')
% xlabel('Tiempo (s)')
% ylabel('Amplitud (V)')
% grid on
% 
% subplot(2,1,2)
% plot(tiempos_wl, wl_ventanas, 'r', 'LineWidth', 1.5)
% title('Waveform Length (ventanas de 0.2s)')
% xlabel('Tiempo (s)')
% ylabel('WL')
% grid on

end