clear;
clc;
close all;




%% 读取

[T32,M32,E32]=read_series(...
"runs/window-l32/series.jsonl");


[T64,M64,E64]=read_series(...
"runs/window-l64/series.jsonl");



%% ======================
% 计算
% ======================




[T32u,M32a,chi32]=analysis(T32,M32,32);


[T64u,M64a,chi64]=analysis(T64,M64,64);

%% M-T

figure

plot(T32u,M32a,'o-')

hold on

plot(T64u,M64a,'s-')

xlabel("Temperature")

ylabel("<M>")

legend("L=32","L=64")

grid on



%% chi-T

figure

plot(T32u,chi32,'o-')

hold on

plot(T64u,chi64,'s-')

xlabel("Temperature")

ylabel("\chi")

legend("L=32","L=64")

grid on



%% Tc

[~,i1]=max(chi32);

[~,i2]=max(chi64);


fprintf("Tc(L32)=%.3f\n",T32u(i1));

fprintf("Tc(L64)=%.3f\n",T64u(i2));