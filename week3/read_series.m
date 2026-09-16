%% ======================
% 快速读取jsonl
% ======================

function [T,M,E]=read_series(filename)


fid=fopen(filename,'r');


T=[];
M=[];
E=[];


while true

    line=fgetl(fid);


    if ~ischar(line)
        break
    end


    s=jsondecode(line);


    T(end+1)=s.T;

    M(end+1)=s.M;

    E(end+1)=s.E;


end


fclose(fid);


T=T';
M=M';
E=E';


end

