% code for calculate CBV/CBF maps of MOLED T2/T2* time series
% Created by Qinqin Yang (qqyang@stu.xmu.edu.cn)
% Time 2024/01/22

clc,clear;close all;

workpath = 'meas_MID02719_FID1377060_a_sage_oled_Dynamic_Charles_Mz\';
fn=[workpath,'slice_006.mat'];

workdir = 'Sub07\';
fn_T1=[workdir,'T1_map.mat'];
fn_AIF=[workdir,'AIF_filter.mat'];

load(fn); % [256   256    12   120] --> [nR,nP,slicen,time points]
load(fn_T1);
load(fn_AIF);

t1_slicen = 19;
slicei = 1;   %1
TR = 1.9;       % uint: s
r1 = 3.6;      % uint: L/mmol/s

maskkey = 0;    % 0: regenrate mask 1: using previous mask
display = 1;

nSamplesMax = 20; % time point of CA injection

% Relaxation time
data_r = rot90(squeeze(results),2);
data_r = rev_tensor(data_r);
if display ==1
    figure;imshow3(data_r,[0 1]);colormap gray;
end

% Mask generation
data1 = mean(data_r(:,:,1:10),3);
if maskkey == 0
    T = 0.05; % need to adjust
    mask = mask_generate(data1,T);
else
    fn_mask = [workdir,'MASK.mat'];
    load(fn_mask);
end

if display ==1
    figure;imshow([data1,mask,data1.*mask],[0 1]);
end

% Load T1
t1data = imresize(T1_map(:,:,t1_slicen)./1000,[256,256]); 
t1data = flip(t1data,2);


if display ==1
    figure;imshow([t1data, mask, t1data.*mask],[0 4]);
end

% Load AIF
AIF = AIF_fit_filter;
plot(AIF);

% T1 and M0 registration
moving = t1data.*mask;
fixed = data1.*mask*4;

transformType = 'affine'; 
optimizer = registration.optimizer.RegularStepGradientDescent;
metric = registration.metric.MeanSquares;

registered = imregister(moving, fixed, transformType, optimizer, metric);

figure;
subplot(1,3,1);
imshow(fixed,[0 4]);
subplot(1,3,2);
imshow(moving,[0 4]);
subplot(1,3,3);
imshowpair(fixed,registered,'falsecolor');

t1data = registered.*mask;
t1data = ones(size(t1data))*1.9.*mask;
R1data = (1./t1data).*mask;

%% CA concentration
S0map=mean(data_r(:,:,1:nSamplesMax),3);
CONCF = sig2conc_oled(data_r,R1data,TR,S0map,mask);

if display ==1
    figure;imshow3(CONCF,[0,0.5]);colormap jet;
end

maxValue = max(AIF);
index = find(AIF == maxValue);

time_in=index-2;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(CONCF,3);
% time=[zeros(1,time_in),[1:(frames-time_in)]*tpres];
time=[0,(1:frames-1)*tpres];

CONCF = imresize(CONCF,[128,128],'nearest');
mask = imresize(mask,[128,128],'nearest');
[ktrans,ve,Fp,Tc]=conc2Ktrans_tissue_homo(CONCF,time,AIF,mask);

%%
figure(21);
subplot(1,4,1);imshow(ktrans,[0 0.1]);title('Kt');colorbar;
subplot(1,4,2);imshow(Fp,[0 1]);title('Fp');colorbar;
subplot(1,4,3);imshow(ve,[0 0.1]);title('Ve');colorbar;
subplot(1,4,4);imshow(Tc,[0 0.6]);title('Tc');colorbar;
colormap jet;
