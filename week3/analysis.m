function [Tlist,Mavg,chi]=analysis(T,M,L)


Tlist=unique(T);


Mavg=zeros(size(Tlist));

chi=zeros(size(Tlist));


for i=1:length(Tlist)


    idx=(T==Tlist(i));


    tempM=M(idx);


    Mavg(i)=mean(tempM);


    chi(i)=L^2/Tlist(i)*...
    (mean(tempM.^2)-mean(tempM)^2);



end

end



