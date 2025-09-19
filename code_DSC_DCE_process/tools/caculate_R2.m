function R2 = caculate_R2(volumes,mask,nSamplesMax)
S0=DSC_cal_S0_mean(volumes,mask,nSamplesMax);

R2=zeros(size(volumes));
[nR,nC,nT]=size(volumes);
ind=find(mask);
k=nR*nC;
R0=1./S0(ind);

for t=1:nT
    Rt=1./volumes(ind+k*(t-1));
    temp=Rt-R0;
    temp(temp<0)=0;
    R2(ind+k*(t-1))=temp;
end

R2=abs(R2);
R2(isnan(R2)) = 0;
R2(isinf(R2)) = 0;
end

