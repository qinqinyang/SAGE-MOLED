clear,clc;
load("Simu_Template.mat");
load("Simu_AIF.mat");

%% DSC R(t) and Ct(t)
rho = 1.04;
Hf = 0.7;
TR = 1;
frames = size(ca,2);
time = 0:TR:frames*TR-1;
t=0:frames-1;
[ww,hh] = size(CBF);
R = zeros(ww,hh,frames);
Ct_dsc = zeros(ww,hh,frames);
P = zeros(ww,hh);
AIF = ca;
for wi = 1:ww
    for hi = 1:hh
        if CBV(wi,hi) > 0
            cbf_temp = CBF(wi,hi)./(100.*60);
            cbv_temp = CBV(wi,hi)./(100);
            R_temp = exp(-cbf_temp.*t./cbv_temp);
            R_temp = R_temp(:);
            Ct_dsc(wi,hi,:) = rho./Hf.*cbf_temp.*filter(R_temp,1,AIF);
        end
    end
end

figure;
imshow3(Ct_dsc,[0,0.6]);colormap jet;
title('Ct(t) of DSC')

%% DCE Ct(t)
TR = 1.9;
time_in=19;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(ca,2);
time=[zeros(1,time_in),[1:(frames-time_in)]*tpres];
Cp = ca';

Ct_dce = zeros(ww,hh,frames);
parfor ii=1:ww
    for jj=1:hh
        if M0(ii,jj) > 0
            Ct_dce(ii,jj,:) = model_extended_tofts_simu(ktrans(ii,jj), ve(ii,jj), vp(ii,jj), Cp, time);
        end
    end
    disp(ii);
end

figure;
imshow3(Ct_dce,[0 0.5]);colormap jet;
title('Ct(t) of DCE')

%%
Ct_dce = single(Ct_dce);
Ct_dsc = single(Ct_dsc);
clearvars -except CBF CBV ktrans M0 t1 t2 t2star ve vp mask Ct_dsc Ct_dce

%% from Ca to T1/T2/T2star
r1 = 3.6; %L*mmol-1*s-1
r2 = 20.4;
r2star = 87.0;

Ct_dce(isnan(Ct_dce))=0;
Ct_dsc(isnan(Ct_dsc))=0;

%% T1
R1_base = 1./t1;
R1_base(isnan(R1_base))=0;
R1_dyn = Ct_dce .* r1 + R1_base;
t1_dyn = 1./R1_dyn;
t1_dyn(isnan(t1_dyn))=0;

%% T2star
R2star_base = 1./t2star;
R2star_base(isnan(R2star_base))=0;
R2star_dyn = Ct_dsc .* r2star + R2star_base;
t2star_dyn = 1./R2star_dyn;
t2star_dyn(isnan(t2star_dyn))=0;

%% T2star
R2_base = 1./t2;
R2_base(isnan(R2_base))=0;
R2_dyn = Ct_dsc .* r2 + R2_base;
t2_dyn = 1./R2_dyn;
t2_dyn(isnan(t2_dyn))=0;

%% 
mask = zeros(512,512);
mask(mean(t1_dyn,3)>0.001 & mean(t2_dyn,3)>0.001 & mean(t2star_dyn,3)>0.001) = 1 ;

t2_dyn = t2_dyn.*mask;
t2star_dyn = t2star_dyn.*mask;
t1_dyn = t1_dyn.*mask;

%%
figure;
imshow3(t1_dyn,[0 3.5]);colormap jet;
figure;
imshow3(t2star_dyn,[0 0.2]);colormap jet;
figure;
imshow3(t2_dyn,[0 0.2]);colormap jet;
clearvars -except t1_dyn t2_dyn t2star_dyn M0 mask
