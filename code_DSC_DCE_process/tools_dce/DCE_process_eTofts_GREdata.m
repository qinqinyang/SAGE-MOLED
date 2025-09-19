% code for calculate CBV/CBF maps of MOLED T2/T2* time series
% Created by Qinqin Yang (qqyang@stu.xmu.edu.cn)
% Time 2024/01/22

clc,clear;
fn='data_In_vivo\Data_GRE\CHEN_XUE_MING_MR0280872\';
dirlist = dir([fn,'T1_VIBE*']);
slicen = 20;
ww = 192;
framen = 35;
TR = 5.9;
data = zeros(ww,ww,slicen,framen);
for framei=1:framen
    dirname = [fn,dirlist(framei).name,'\'];
    filelist = dir([dirname,'*IMA']);
    for slicei = 1:slicen
        filename = [dirname,filelist(slicei).name];
        data(:,:,slicei,framei)=double(dicomread(filename));
    end
    disp(['load frame ',num2str(framei)]);
end

slicei = 5;
r1 = 3.6;      % uint: L/mmol/s
r2 = 20.4;
AIFkey = 1;     % 0: manual selection 1: using previous AIF
maskkey = 0;    % 0: regenrate mask 1: using previous mask
display = 1;

nSamplesMax = 6; % time point of CA injection

% Relaxation time
data_r = squeeze(data(:,:,slicei,:));

if display ==1
    figure;imshow3(data_r,[0 300]);colormap gray;
end

% Mask generation
data1 = mean(data_r(:,:,1:10),3);
if maskkey == 0
    T = 0.1; % need to adjust
    mask = mask_generate(data1,T);
else
    fn_mask = 'MASK.mat';
    load(fn_mask);
end

if display ==1
    figure;imshow([data1,mask,data1.*mask],[0 1]);
end

%% Load T1
fndcm='data_In_vivo\Data_GRE\T1_IMAGES_B1CORR_0025\';
ww=192;
wwslicen=4;
t1data=zeros(192,192,20);
dcmlist=dir([fndcm,'*IMA']);
for dcmi=1:length(dcmlist)
    fntemp=[fndcm,dcmlist(dcmi).name];
    t1data(:,:,dcmi)=double(dicomread(fntemp))./1000;
end
mask = double(mask);
t1data = t1data(:,:,wwslicen);
R1data = (1./t1data).*mask;

if display ==1
    figure;imshow([t1data, mask, t1data.*mask],[0 3]);
end

%% CA concentration
alpha = deg2rad(15);
TRi = 5.1 * 1e-3;
S0map=mean(data_r(:,:,1:nSamplesMax),3);
CONCF = sig2conc_dce(data_r,R1data,alpha,TRi,S0map,mask);

if display ==1
    figure;imshow3(CONCF,[0,0.3]);colormap jet;
end

time_in=6;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(CONCF,3);
time=[zeros(1,time_in),[1:(frames-time_in)]*tpres];

%% AIF calculation
if AIFkey == 0
    showtime = 8;
    slice_aif = 11;
    data_aif = squeeze(data(:,:,slice_aif,:));
    [AIF_fit,aifx,aify] = AIF_selection_dce2(data_aif,R1data,alpha,TRi,showtime);
elseif AIFkey == 1
    AIF_fit=SAIF_p(time);
else
    fn_aif = 'data_In_vivo\Data_GRE\AIF_DCE.mat';
    load(fn_aif);
end
AIF = AIF_fit;
plot(AIF);
%% CBV CBV MTT
CONCF = imresize(CONCF,[128,128],'nearest');
mask = imresize(mask,[128,128],'nearest');
[ktrans,ve,vp]=conc2Ktrans_eTofts(CONCF,time,AIF,mask);
% [ktrans,vp]=conc2Ktrans_patlak(CONCF,time,AIF,mask);
% [ktrans,ve]=conc2Ktrans_tofts(CONCF,time,AIF,mask);

%%
figure(11);
subplot(1,3,1);imshow(ktrans,[0 0.4]);title('Kt');colorbar;
subplot(1,3,2);imshow(vp,[0 0.4]);title('Vp');colorbar;
subplot(1,3,3);imshow(ve,[0 1]);title('Ve');colorbar;
colormap jet;
