clear;clc;


filename='runs/ramp/spins.jsonl';

fid=fopen(filename,'r');


data={};

while ~feof(fid)

    line=fgetl(fid);

    if ischar(line)

        data{end+1}=jsondecode(line);

    end

end

fclose(fid);



target_T=[1.8 2.3 3.0];


figure;

for k=1:length(target_T)


    T0=target_T(k);


    % 找最接近温度的数据
    idx=[];

    for i=1:length(data)

        if abs(data{i}.T-T0)<1e-6

            idx=i;
            break;

        end

    end


    if isempty(idx)

        warning("没有找到 T=%f",T0);

        continue

    end


    spins=data{idx}.spins;


    % 一维转二维
    spins=reshape(spins,64,64);


    subplot(1,3,k)

    imagesc(spins)

    colormap(gray)

    axis equal

    axis tight


    title(sprintf('T=%.2f',T0))

    xlabel('x')
    ylabel('y')


end