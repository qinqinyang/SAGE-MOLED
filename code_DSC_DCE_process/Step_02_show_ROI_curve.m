clear,clc;

workpath = 'meas_MID00371_FID1534392_a_sage_oled_Dynamic_Charles_T2T2star\';
fn=[workpath,'slice_006.mat'];
fn_AIF = 'AIF_filter.mat';

load(fn);
load(fn_AIF);
AIF = AIF_fit';

framen = size(results,3);
AIF = AIF(1:framen);

maxValue = max(AIF);
index = find(AIF == maxValue);

rho = 1.04;
Hf = 0.7;

TR = 1.9;
r2s = 87;      % uint: L/mmol/s T2star
r2 = 20.4;     % uint: L/mmol/s T2
nSamplesMax = index-10; % time point of CA injection

% Mask generation for T2
data_r = results(:,:,:,1);
data_r = rev_tensor(data_r);
data_t2 = rot90(data_r,2);

data1 = mean(data_t2(:,:,5:nSamplesMax-1),3);
T = 0.1;
mask_t2 = mask_generate(data1,T);
figure;imshow([data1,mask_t2,data1.*mask_t2],[0 0.2]);

% Mask generation for T2star
data_r = results(:,:,:,2);
data_r = rev_tensor(data_r);
data_t2star = rot90(data_r,2);
% Ca(t) for T2
[Ct_t2,S0] = caculate_Ct(data_t2,mask_t2,nSamplesMax,r2);
figure;imshow3(Ct_t2,[0,0.6]);colormap jet;

% Ca(t) for T2star
[Ct_t2star,S0] = caculate_Ct(data_t2star,mask_t2,nSamplesMax,r2s);
figure;imshow3(Ct_t2star,[0,0.6]);colormap jet;

clearvars -except data_t2 data_t2star Ct_t2 Ct_t2star index framen
%%
dynamic_t2=zeros(framen,1);
dynamic_t2star=zeros(framen,1);
dynamic_C_t2=zeros(framen,1);
dynamic_C_t2star=zeros(framen,1);

temp = Ct_t2star(:,:,index+5);
figure;
imagesc(temp,[0 0.3]);colormap jet;axis image;
hfh=imfreehand();
ROI_mask=hfh.createMask();

for i=1:framen
    temp_t2=data_t2(:,:,i);
    temp_t2star=data_t2star(:,:,i);
    temp_t2_C=Ct_t2(:,:,i);
    temp_t2star_C=Ct_t2star(:,:,i);
    dynamic_t2(i,1)=mean(temp_t2(ROI_mask==1));
    dynamic_t2star(i,1)=mean(temp_t2star(ROI_mask==1));
    dynamic_C_t2(i,1)=mean(temp_t2_C(ROI_mask==1));
    dynamic_C_t2star(i,1)=mean(temp_t2star_C(ROI_mask==1));
end

dynamic_C_t2=smooth(dynamic_C_t2,'moving');

x=[1:1:framen];
figure(112);
subplot(4,1,1);
plot(x,dynamic_t2,'LineWidth',3);
ylim([0 0.08]);
xlim([0 framen]);
set(gca,'FontSize',20);
title('T2','FontSize',20);xlabel('Measurements','FontSize',20);ylabel('T2 value (s)','FontSize',20);
grid on;

subplot(4,1,2);
plot(x,dynamic_C_t2,'LineWidth',3);
ylim([0 1]);
xlim([0 framen]);
set(gca,'FontSize',20);
title('CA concentration (t2)','FontSize',20);xlabel('Measurements','FontSize',20);ylabel('Concentration (mmol)','FontSize',20);
grid on;

subplot(4,1,3);
plot(x,dynamic_t2star,'LineWidth',3);
ylim([0 0.08]);
xlim([0 framen]);
set(gca,'FontSize',20);
title('T2star','FontSize',20);xlabel('Measurements','FontSize',20);ylabel('T2star value (s)','FontSize',20);
grid on;

subplot(4,1,4);
plot(x,dynamic_C_t2star,'LineWidth',3);
ylim([0 1]);
xlim([0 framen]);
set(gca,'FontSize',20);
title('CA concentration (t2star)','FontSize',20);xlabel('Measurements','FontSize',20);ylabel('Concentration (mmol)','FontSize',20);
grid on;