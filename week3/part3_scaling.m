clear;
clc;
close all;


%% =========================
% Part3 finite size scaling
% =========================


% system size
L = [16 32 64];


% Tc obtained from susceptibility peak
Tc_L = [2.20 2.25 2.35];


%% 1/L

x = 1./L;


%% linear fitting

p = polyfit(x,Tc_L,1);


a = p(1);
Tc = p(2);


fprintf("Estimated Tc = %.4f\n",Tc);


%% fitted line

xfit = linspace(0,max(x)*1.2,100);

yfit = polyval(p,xfit);



%% plot

figure;

plot(x,Tc_L,'o',...
    'MarkerSize',8,...
    'LineWidth',2);

hold on;

plot(xfit,yfit,...
    'LineWidth',2);


xlabel('1/L');

ylabel('T_c(L)');

title('Finite Size Scaling of Critical Temperature');

legend('Simulation',...
       'Linear fit');


grid on;

