%% =========================
% 读取 jsonl 数据函数
% =========================

function data = read_jsonl(filename)

fid=fopen(filename,'r');

data=[];

while ~feof(fid)

    line=fgetl(fid);

    if ischar(line)

        tmp=jsondecode(line);

        data=[data; tmp];

    end

end

fclose(fid);

end

