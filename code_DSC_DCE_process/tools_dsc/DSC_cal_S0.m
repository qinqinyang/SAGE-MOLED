function S0map=DSC_cal_S0(volumes,mask,nSamplesMax)
% ciclo=true;
% pos=3;
% while ciclo
%     mean_val=mean(mean_signal(s,1:pos));
%     if abs((mean_val-mean_signal(s,pos+1))/mean_val)<thresh
%         pos=pos+1;
%     else
%         ciclo=false;
%         pos=pos-1;
%     end
%     if pos==nSamplesMax
%         ciclo=false;
%         pos=pos-1;
%     end
% end
% S0map=mask.*mean(volumes(:,:,1:pos),3);

nSamplesMin=2;
thresh=0.1;

[nR,nC,nT]=size(volumes);
mean_signal=zeros(1,nT);

for t=1:nT
    indMask=find(mask(:,:));
    mean_signal(t)=mean(volumes(indMask+(nR*nC)*(t-1)));
end

ciclo=true;
pos=nSamplesMin;
while ciclo
    mean_val=mean(mean_signal(1:pos));
    disp(pos);
    disp(mean_val);
    if abs((mean_val-mean_signal(pos+1))/mean_val)<thresh
        pos=pos+1;
    else
        ciclo=false;
        pos=pos-1; 
    end
    if pos==nSamplesMax
        ciclo=false;
        pos=pos-1;
    end
end

S0map=mask.*mean(volumes(:,:,1:pos),3);

end