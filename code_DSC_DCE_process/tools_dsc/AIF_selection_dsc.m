function [aif_cuv,x,y] = AIF_selection_dsc(data,showtime,nSamples)

nSamplesMax = nSamples; % time point of CA injection

data_r = data;
[nR,nP,nT] = size(data_r);

mask = ones(nR,nP);
[Ct,S0] = caculate_Ct_aif(data_r,mask,nSamplesMax);

%%
figure;imagesc(squeeze(Ct(:,:,showtime)),[0 80]);colormap jet;axis image;
% figure;imagesc(squeeze(data(:,:,2)),[0 0.2]);colormap jet;axis image;
[x,y]=getpts;

points=size(data_r,3);
aif_data=zeros(points, length(x));
for i=1:length(x)
    idx=round(y(i));
    idy=round(x(i));
    aif_data(:,i)=squeeze(Ct(idx,idy,:));
end

aif_cuv=mean(aif_data,2);
r=0.493;
q=2.616;

aif_cuv=(-r+sqrt(r^2+4*q.*aif_cuv))./(2*q);

%%
figure;
plot(aif_cuv);

x = round(x);
y = round(y);
end

function [conc,S0] = caculate_Ct_aif(volumes,mask,nSamplesMax)
% volumes = volumes*1000; % s--->ms
S0=DSC_cal_S0_mean(volumes,mask,nSamplesMax);

conc=zeros(size(volumes));
[nR,nC,nT]=size(volumes);
ind=find(mask);
k=nR*nC;
R0=1./S0;

for t=1:nT
    Rt=1./volumes(:,:,t);
    temp=(Rt-R0).*mask;
    temp(temp<0)=0;
    conc(:,:,t)=temp;
end

% conc=real(conc);
conc(isnan(conc)) = 0;
conc(isinf(conc)) = 0;
end