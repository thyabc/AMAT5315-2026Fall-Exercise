clear;
clc;
close all;


%% 读取 series.jsonl

filename = "../runs/ramp/series.jsonl";

lines = readlines(filename);

% 删除空行
lines(lines=="") = [];

N = length(lines);

T = zeros(N,1);
M = zeros(N,1);
E = zeros(N,1);


for i = 1:N

    data = jsondecode(lines(i));

    T(i)=data.T;
    M(i)=data.M;
    E(i)=data.E;

end



%% 对每个温度求平均

T_unique = unique(T);


M_avg=zeros(size(T_unique));


for i=1:length(T_unique)

    index = T == T_unique(i);

    M_avg(i)=mean(abs(M(index)));

end



%% 绘图


figure;

plot(T_unique,M_avg,...
    '-o',...
    'LineWidth',1.5,...
    'MarkerSize',5);


xlabel('Temperature T');

ylabel('Magnetization |M|');


title('Magnetization vs Temperature');


grid on;


set(gca,'FontSize',12);