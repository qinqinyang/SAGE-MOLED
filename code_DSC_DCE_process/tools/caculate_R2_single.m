function R2 = caculate_R2_single(volumes,nSamplesMax)

S0=mean(volumes(3:nSamplesMax),1);

R2=zeros(size(volumes));
[nT]=size(volumes);
R0=1./S0;

for t=1:nT
    Rt=1./volumes(t);
    temp=Rt-R0;
    temp(temp<0)=0;
    R2(t)=temp;
end

R2=abs(R2);
R2(isnan(R2)) = 0;
R2(isinf(R2)) = 0;
end
