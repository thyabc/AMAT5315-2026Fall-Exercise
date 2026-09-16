clear;
clc;
close all;


filename="../runs/ramp/series.jsonl";


lines = readlines(filename);

% 删除空行
lines(lines=="") = [];

N = length(lines);

T=zeros(N,1);
E=zeros(N,1);



for i=1:N

    data=jsondecode(lines(i));


    T(i)=data.T;

    E(i)=data.E;

end



%% 平均


T_unique=unique(T);


E_avg=zeros(size(T_unique));


for i=1:length(T_unique)

    index=T==T_unique(i);


    E_avg(i)=mean(E(index));

end



%% plot


figure;


plot(T_unique,E_avg,...
    '-o',...
    'LineWidth',1.5);


xlabel("Temperature T");

ylabel("Energy E");


title("Energy vs Temperature");


grid on;


set(gca,'FontSize',12);