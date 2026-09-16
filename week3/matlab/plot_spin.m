clear;
clc;
close all;


filename="../runs/ramp/spins.jsonl";


lines=readlines(filename);



%% 选择第一个snapshot

data=jsondecode(lines(1));



L=data.L;


spin=data.spins;


spin=reshape(spin,[L,L]);



figure;


imagesc(spin);


axis equal tight;


colormap(gray);


colorbar;


title(sprintf("Spin configuration T=%.2f",data.T));


xlabel("x");

ylabel("y");
