function[mtx_coeficientes]=f_cc(segment,fs)
    % 1. Parámetros de ventaneo (250 ms para mejor resolución)
    ventana_tiempo = 0.25;         % 250 ms por ventana
    muestras_ventana = round(ventana_tiempo * fs); % 25 muestras
    solapamiento = round(muestras_ventana * 0.5);  % 12 muestras (50%)
    
    % 2. Calcular número de ventanas
    num_ventanas = floor((length(segment) - muestras_ventana)/...
                    (muestras_ventana - solapamiento) + 1);
    coeficientes = zeros(num_ventanas, 13); % Matriz para 13 coeficientes

    % 3. Procesamiento por ventanas
    for i = 1:num_ventanas
        % Extraer ventana
        inicio = (i-1)*(muestras_ventana - solapamiento) + 1;
        fin = inicio + muestras_ventana - 1;
        ventana = segment(inicio:fin);
        
        % Aplicar ventana de Hamming
        ventana = ventana .* hamming(muestras_ventana)';
        
        % FFT optimizada
        N_fft = 2^nextpow2(muestras_ventana); % Tamaño FFT = 32 (para 25 muestras)
        espectro = fft(ventana, N_fft);
        
        % Espectro logarítmico (solo frecuencias positivas)
        log_espectro = log(abs(espectro(1:N_fft/2+1)) + eps);
        
        % Cálculo cepstral
        cepstrum = ifft(log_espectro);
        
        % Extraer 13 coeficientes (excluyendo el coeficiente 0)
        coeficientes(i,:) = real(cepstrum(2:14))'; 
    end
    mtx_coeficientes=coeficientes(:)';
end