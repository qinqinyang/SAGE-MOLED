clear,clc;

workpath = 'meas_MID02719_FID1377060_a_sage_oled_Dynamic_Charles_T2T2star\';
fn=[workpath,'slice_007.mat'];
fn_AIF = 'AIF_filter.mat';

load(fn);
load(fn_AIF);
AIF = AIF_fit';

framen = size(results,3);
AIF = AIF(1:framen);
maxValue = max(AIF);
index = find(AIF == maxValue);
nSamplesMax = index-5; % time point of CA injection

rho = 1.04;
Hf = 0.7;

TR = 1.9;
r2s = 87;      % uint: L/mmol/s T2star
r2 = 20.4;     % uint: L/mmol/s T2
frames = size(AIF,2);

%% Mask generation for T2
data_r = results(:,:,:,1);
data_r = rev_tensor(data_r);
data_t2 = rot90(data_r,2);

data1 = mean(data_t2(:,:,1:nSamplesMax),3);
T = 0.1;
mask_t2 = mask_generate(data1,T);
figure;imshow([data1,mask_t2,data1.*mask_t2],[0 0.2]);

%% Mask generation for T2star
data_r = results(:,:,:,2);
data_r = rev_tensor(data_r);
data_t2star = rot90(data_r,2);
%% Ca(t) for T2
[Ct_t2,S0] = caculate_Ct(data_t2,mask_t2,nSamplesMax,r2);
figure;imshow3(Ct_t2,[0,0.6]);colormap jet;

%% Ca(t) for T2star
[Ct_t2star,S0] = caculate_Ct(data_t2star,mask_t2,nSamplesMax,r2s);
figure;imshow3(Ct_t2star,[0,0.6]);colormap jet;

%% cal CBV CBF
type = 'cSVD';
options.display = 1;
options.nT = framen;
options.deconv.cSVD.threshold = 0.01;
options.deconv.cSVD.residual = 1;
options.deconv.SVD.residual = 1;
options.nR = 256;
options.nC = 256;
options.waitbar = 1;
options.tr = 1.9;
options.time=0:options.tr:(options.nT-1)*options.tr;

[CBV_t2]=DSC_mri_cbv(Ct_t2,AIF,mask_t2,TR,Hf,rho);
[CBV_t2_corrected,K1_map,K2_map,K1_CV_map,K2_CV_map]=DSC_mri_cbv_lc(Ct_t2,AIF,mask_t2,32,options);
[CBF_t2]=DSC_mri_cbf_leakage(Ct_t2,AIF,mask_t2,TR,Hf,rho,type,options);

[CBV_t2star]=DSC_mri_cbv(Ct_t2star,AIF,mask_t2,TR,Hf,rho);
[CBV_t2star_corrected,K1_map,K2_map,K1_CV_map,K2_CV_map]=DSC_mri_cbv_lc(Ct_t2star,AIF,mask_t2,32,options);
[CBF_t2star]=DSC_mri_cbf_leakage(Ct_t2star,AIF,mask_t2,TR,Hf,rho,type,options);

%%
hotIronColors = [
    0, 0, 0;     
    0, 0.5, 1;   
    0, 0.6, 0;   
    1, 1, 0;     
    1, 0, 0;    
    1, 1, 1;    
];

hotIronMap = interp1(linspace(0, 1, size(hotIronColors, 1)), hotIronColors, linspace(0, 1, 256));

figure(11);
subplot(1,3,1);imshow(CBV_t2,[0 10]);title('CBV T2');colorbar;
subplot(1,3,2);imshow(CBV_t2_corrected*500,[0 10]);title('CBV T2 corrected');colorbar;
subplot(1,3,3);imshow(CBF_t2,[0 150]);title('CBF T2');colorbar;
colormap(hotIronMap);

figure(22);
subplot(1,3,1);imshow(CBV_t2star,[0 10]);title('CBV T2star');colorbar;
subplot(1,3,2);imshow(CBV_t2star_corrected*200,[0 10]);title('CBV T2star corrected');colorbar;
subplot(1,3,3);imshow(CBF_t2star,[0 150]);title('CBF T2star');colorbar;
colormap(hotIronMap);