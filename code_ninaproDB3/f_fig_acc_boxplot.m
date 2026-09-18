clc; close;
% Si ya tienes las matrices que exportaste a CSV (sujetos × pipelines):
acc_db1 = readmatrix('resultados_accuracy_ninapro_db1.csv','Range','H2:H28');  % col LDA_MAV
acc_db3 = readmatrix('resultados_accuracy_ninapro_db3.csv','Range','H2:H12');
acc_dba = readmatrix('resultados_accuracy_capgmyo_dba.csv','Range','H2:H19');

f_fig_persubject_accuracy_v3(acc_db1, acc_db3, acc_dba, 'Fig2_persubject');


% T1 = readtable('resultados_accuracy_ninapro_db1.csv');
% T3 = readtable('resultados_accuracy_ninapro_db3.csv');
% Ta = readtable('resultados_accuracy_capgmyo_dba.csv');
% 
% fig = f_fig_persubject_accuracy_v3(T1.LDA_MAV, T3.LDA_MAV, Ta.LDA_MAV, 'Fig2_persubject');
% %fig = f_fig_persubject_accuracy_revMS(T1.LDA_MAV, T3.LDA_MAV, Ta.LDA_MAV, 'Fig2_persubject');