function fig = f_fig_persubject_accuracy(acc_db1, acc_db3, acc_dba, nombreSalida)
%F_FIG_PERSUBJECT_ACCURACY  Figura 2: accuracy por sujeto (MAV+LDA).
%
%   fig = f_fig_persubject_accuracy(acc_db1, acc_db3, acc_dba)
%   fig = f_fig_persubject_accuracy(acc_db1, acc_db3, acc_dba, 'Fig2')
%
%   ENTRADAS
%     acc_db1 : vector accuracy (%) por sujeto - Ninapro DB1  (27)
%     acc_db3 : vector accuracy (%) por sujeto - Ninapro DB3  (11)
%     acc_dba : vector accuracy (%) por sujeto - CapgMyo DB-a (18)
%     nombreSalida : (opcional) nombre base de salida. Def. 'Fig2_persubject'.
%
%   Escala de grises, ancho de columna LNCS (11.7 cm). Exporta TIFF 600 dpi + EPS.
%
%   CORRECCIONES (v3):
%     - Bordes de caja y bigotes en negro grueso (LineWidth 1.2).
%     - Mediana redibujada como linea negra gruesa (boxchart la traza fina).
%     - Etiquetas del eje X en DOS lineas (nombre + n) via XTickLabel, para que
%       el "(n=..)" no se solape con el nombre del dataset.

    if nargin < 4 || isempty(nombreSalida)
        nombreSalida = 'Fig2_persubject';
    end

    % --- Orden FIJO: 1=DB1, 2=DB3, 3=CapgMyo ---
    datos     = {acc_db1(:), acc_db3(:), acc_dba(:)};
    nombres   = {'Ninapro DB1', 'Ninapro DB3', 'CapgMyo DB-a'};
    posiciones = [1 2 3];

    colRelleno = [0.851 0.851 0.851];
    colPunto   = [0.251 0.251 0.251];
    LW_BOX     = 1.2;    % grosor de borde de caja y bigotes
    LW_MED     = 1.8;    % grosor de la linea de mediana

    fig = figure('Units','centimeters', 'Position',[2 2 11.7 7.4], ...
                 'Color','w', 'PaperUnits','centimeters', ...
                 'PaperSize',[11.7 7.4], 'PaperPosition',[0 0 11.7 7.4]);
    ax = axes(fig); hold(ax,'on');

    % --- Una caja por grupo, en su posicion explicita, con borde grueso ---
    for g = 1:3
        y = datos{g};
        boxchart(ax, repmat(posiciones(g), numel(y), 1), y, ...
                 'BoxWidth', 0.55, ...
                 'BoxFaceColor', colRelleno, 'BoxFaceAlpha', 1, ...
                 'BoxEdgeColor', 'k', 'LineWidth', LW_BOX, ...
                 'WhiskerLineColor', 'k', 'MarkerStyle', 'none');
    end

    % --- Mediana redibujada en negro grueso sobre cada caja ---
    % (boxchart no expone el grosor de la mediana; se dibuja manualmente)
    for g = 1:3
        md = median(datos{g});
        w  = 0.55/2;                      % media anchura de la caja
        plot(ax, posiciones(g)+[-w w], [md md], 'k-', 'LineWidth', LW_MED);
    end

    % --- Puntos individuales con jitter (stream local) ---
    s = RandStream('mt19937ar', 'Seed', 42);
    for g = 1:3
        y = datos{g};
        x = posiciones(g) + 0.055 * randn(s, numel(y), 1);
        scatter(ax, x, y, 13, 'filled', ...
                'MarkerFaceColor', colPunto, 'MarkerFaceAlpha', 0.75, ...
                'MarkerEdgeColor', 'w', 'LineWidth', 0.4);
    end

    % --- Media (rombo blanco) y etiqueta SD a la derecha ---
    for g = 1:3
        y = datos{g};
        scatter(ax, posiciones(g), mean(y), 40, 'd', 'filled', ...
                'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 1.3);
        text(ax, posiciones(g) + 0.30, median(y), sprintf('SD %.1f', std(y)), ...
             'FontSize', 7.2, 'FontAngle','italic', ...
             'Color', [0.2 0.2 0.2], 'HorizontalAlignment','left');
    end

    % --- Eje X: etiquetas de DOS lineas (nombre + n) ---
    % Se construye cada etiqueta como cell multilinea; asi el "(n=..)" queda
    % DEBAJO del nombre y no se solapa.
    etiquetas = cell(1,3);
    for g = 1:3
        etiquetas{g} = sprintf('%s\\newline(n=%d)', nombres{g}, numel(datos{g}));
    end
    set(ax, 'XTick', posiciones, 'XTickLabel', etiquetas, ...
            'TickLabelInterpreter', 'tex');
    ax.XAxis.FontSize = 8;

    ylabel(ax, 'Per-subject accuracy (%)', 'FontSize', 8.5);
    ylim(ax, [18 104]);
    xlim(ax, [0.45 3.85]);
    set(ax, 'FontSize', 8, 'Box','off', 'TickDir','out', ...
            'YGrid','on', 'GridAlpha', 0.15, 'LineWidth', 0.75);

    % --- Leyenda con handles ficticios ---
    hL1 = patch(ax, NaN, NaN, colRelleno, 'EdgeColor','k', 'LineWidth', LW_BOX);
    hL2 = scatter(ax, NaN, NaN, 40, 'd', 'filled', ...
                  'MarkerFaceColor','w', 'MarkerEdgeColor','k', 'LineWidth',1.3);
    hL3 = scatter(ax, NaN, NaN, 13, 'filled', 'MarkerFaceColor', colPunto);
    legend(ax, [hL1 hL2 hL3], ...
           {'IQR (box), median (line)', 'Mean', 'Subject'}, ...
           'Location','southwest', 'FontSize', 6.8, 'Box','on');

    exportgraphics(fig, [nombreSalida '.tiff'], 'Resolution', 600);
    exportgraphics(fig, [nombreSalida '.eps'],  'ContentType', 'vector');
    fprintf('Figura exportada: %s.tiff y %s.eps\n', nombreSalida, nombreSalida);

    % --- Resumen para verificacion cruzada ---
    fprintf('\n%-14s %4s %7s %7s %8s\n','Dataset','n','Media','SD','Rango');
    for g = 1:3
        d = datos{g};
        fprintf('%-14s %4d %7.2f %7.2f  %2.0f-%.0f\n', ...
                nombres{g}, numel(d), mean(d), std(d), min(d), max(d));
    end
end
