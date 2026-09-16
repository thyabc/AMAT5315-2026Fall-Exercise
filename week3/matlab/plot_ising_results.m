clc;
clear;
close all;


%% ===============================
% 读取 series.jsonl
% ===============================

series_file = 'runs/ramp/series.jsonl';

fid = fopen(series_file,'r');

data = [];

while ~feof(fid)

    line = fgetl(fid);

    if ischar(line)

        tmp = jsondecode(line);

        data = [data; 
            tmp.T, tmp.M, tmp.E, tmp.sweep];

    end

end

fclose(fid);


T = data(:,1);
M = data(:,2);
E = data(:,3);
sweep = data(:,4);



%% ===============================
% 每个温度求平均
% ===============================

T_unique = unique(T);

M_avg = zeros(size(T_unique));
E_avg = zeros(size(T_unique));


for i=1:length(T_unique)

    index = T==T_unique(i);

    M_avg(i)=mean(abs(M(index)));

    E_avg(i)=mean(E(index));

end



%% ===============================
% Figure 1
% Energy vs Temperature
% ===============================


figure;

plot(T_unique,E_avg,...
    '-o',...
    'LineWidth',1.5,...
    'MarkerSize',6);


xlabel('Temperature T');
ylabel('Energy E');

title('Energy vs Temperature');

grid on;


saveas(gcf,'Energy_vs_Temperature.png');




%% ===============================
% Figure 2
% Magnetization
% ===============================


figure;

plot(T_unique,M_avg,...
    '-o',...
    'LineWidth',1.5,...
    'MarkerSize',6);


xlabel('Temperature T');

ylabel('Magnetization |M|');

title('Magnetization vs Temperature');

grid on;


hold on

% 标出理论Tc

Tc=2.26919;

xline(Tc,'--r',...
    'T_c=2.269');


saveas(gcf,...
    'Magnetization_vs_Temperature.png');





%% ===============================
% 读取 spins.jsonl
% ===============================


spin_file='runs/ramp/spins.jsonl';


fid=fopen(spin_file,'r');


frames={};


while ~feof(fid)

    line=fgetl(fid);

    if ischar(line)

        tmp=jsondecode(line);

        frames{end+1}=tmp;

    end

end


fclose(fid);



%% ===============================
% 找三个温度
% ===============================


target_T=[1.8 2.3 3.0];



figure;


for k=1:3


    best=1;
    mindiff=999;


    for i=1:length(frames)

        diff=abs(frames{i}.T-target_T(k));

        if diff<mindiff

            mindiff=diff;
            best=i;

        end

    end


    frame=frames{best};


    L=frame.L;


    spin_string=frame.spins;


    spin=zeros(L,L);


    for i=1:L

        for j=1:L

            index=(i-1)*L+j;

            if spin_string(index)=='1'

                spin(i,j)=1;

            else

                spin(i,j)=-1;

            end

        end

    end



    subplot(1,3,k)

    imagesc(spin);

    colormap(gray);

    axis equal tight;


    title(sprintf('T = %.2f',frame.T));


    xlabel('x');
    ylabel('y');


end



sgtitle('Spin Configuration');


saveas(gcf,...
    'Spin_configuration_three_T.png');
