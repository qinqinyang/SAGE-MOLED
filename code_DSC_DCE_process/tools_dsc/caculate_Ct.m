function [conc,S0] = caculate_Ct(volumes,mask,nSamplesMax,r2)
S0=DSC_cal_S0_mean(volumes,mask,nSamplesMax);

conc=zeros(size(volumes));
[nR,nC,nT]=size(volumes);
ind=find(mask);
k=nR*nC;
R0=1./S0(ind);

for t=1:nT
    Rt=1./volumes(ind+k*(t-1));
    temp=Rt-R0;
    temp(temp<0)=0;
    conc(ind+k*(t-1))=temp./r2;
end

conc=real(conc);
conc(isnan(conc)) = 0;
conc(isinf(conc)) = 0;
end

