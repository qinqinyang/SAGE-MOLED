% Created by Qinqin Yang
% Time 2024/01/22

clear,clc;close all;

fn_T2T2star = 'meas_MID02719_FID1377060_a_sage_oled_Dynamic_Charles_T2T2star\';
fn=[fn_T2T2star,'slice_006.mat'];

fn_DCE = 'DCE_eToft_AIF_01_T1.mat';
fn_AIF='AIF_filter.mat';

load(fn);
load(fn_DCE);
load(fn_AIF);

AIF = AIF_fit_filter';
[Nx,Ny,Nt,~] = size(results);

AIF = AIF(1:Nt);
maxValue = max(AIF);
index = find(AIF == maxValue);
nSamplesMax = index-9;

r2star_p = 87.0;
r2_p = 20.4;

rho = 1.04;
Hf = 0.7;

TR = 1.9; % s

options.display = 1;
options.nT = Nt;
options.deconv.cSVD.threshold = 0.01;
options.deconv.cSVD.residual = 1;
options.deconv.SVD.residual = 1;
options.nR = Nx;
options.nC = Ny;
options.waitbar = 1;
options.tr = TR;

% colormap
hotIronColors = [
    0, 0, 0;
    0, 0.5, 1;
    0, 0.6, 0;
    1, 1, 0;
    1, 0, 0;
    1, 1, 1;
    ];

hotIronMap = interp1(linspace(0, 1, size(hotIronColors, 1)), hotIronColors, linspace(0, 1, 256));

% show AIF
figure(1);
hct = 0.3;
AIF = AIF./(1-hct);
plot(AIF);title('AIF');

% show ktrans vp ve
figure(2);
subplot(1,3,1);imshow(ktrans,[0 0.3]);title('Kt');colorbar;
subplot(1,3,2);imshow(vp,[0 0.02]);title('Vp');colorbar;
subplot(1,3,3);imshow(ve,[0 0.3]);title('Ve');colorbar;
colormap jet;

%% Load T2/T2star and generate Mask from T2
data_r = results(:,:,:,1);  % T2 map
data_r = rev_tensor(data_r);
data_t2 = rot90(data_r,2);

data_r = results(:,:,:,2); % T2star map
data_r = rev_tensor(data_r);
data_t2star = rot90(data_r,2);

temp = mean(data_t2(:,:,2:nSamplesMax-1),3);
T = 0.1;
mask = mask_generate(temp,T);
figure(3);imshow([temp,mask,temp.*mask],[0 0.2]);title('Mask');

%% T2/T2star to R2/R2star and Deconv

data_R2star = caculate_R2(data_t2star,mask,nSamplesMax);

figure(4);imshow3(data_R2star,[0 30]);colormap jet;title('R2star');

data_R2 = caculate_R2(data_t2,mask,nSamplesMax);

figure(5);imshow3(data_R2,[0 10]);colormap jet;title('R2');

%% LSQ for r2_e and r2tar_e
maxValue = max(AIF);
index = find(AIF == maxValue);
time_in=index-1;
tpres=TR/60;

ktrans_mask = ktrans.*mask;
ve_mask = ve.*mask;
vp_mask = vp.*mask;
time=[zeros(1,time_in),[1:(Nt-time_in)]*tpres];

%% for T2star
[r2p_data_t2star, r2e_data_t2star] = r2eMap_ET(data_R2star, ktrans_mask, ve_mask, vp_mask, time, AIF);

figure(6);
subplot(1,2,1);
imshow(r2p_data_t2star,[0 100]);title('r2p-data T2star');colormap jet;
subplot(1,2,2);
imshow(r2e_data_t2star,[0 100]);title('r2e-data T2star');colormap jet;

% Leakage correction for T2star
Ct_corr_t2star = zeros(Nx,Ny,Nt);
for wi=1:Nx
    for hi=1:Ny
        if mask(wi,hi)
            if r2e_data_t2star(wi,hi)>0
                Kt_temp = ktrans_mask(wi,hi);
                ve_temp = ve_mask(wi,hi);
                Dce_term  = model_extended_tofts_dsc_ex(r2e_data_t2star(wi,hi),Kt_temp, ve_temp, AIF, time);
                Ct_corr_t2star(wi,hi,:) = (squeeze(data_R2star(wi,hi,:))-Dce_term)./r2star_p;
            else
                Ct_corr_t2star(wi,hi,:) = squeeze(data_R2star(wi,hi,:))./r2star_p;
            end
        end
    end
end

%% for T2
[r2p_data_t2, r2e_data_t2] = r2eMap_ET(data_R2, ktrans_mask, ve_mask, vp_mask, time, AIF);

figure(7);
subplot(1,2,1);
imshow(r2p_data_t2,[0 100]);title('r2p-data T2');colormap jet;
subplot(1,2,2);
imshow(r2e_data_t2,[0 100]);title('r2e-data T2');colormap jet;

% Leakage correction for T2
Ct_corr_t2 = zeros(Nx,Ny,Nt);
for wi=1:Nx
    for hi=1:Ny
        if mask(wi,hi)
            if r2e_data_t2(wi,hi)>0
                Kt_temp = ktrans_mask(wi,hi);
                ve_temp = ve_mask(wi,hi);
                Dce_term  = model_extended_tofts_dsc_ex(r2e_data_t2(wi,hi),Kt_temp, ve_temp, AIF, time);
                Ct_corr_t2(wi,hi,:) = (squeeze(data_R2(wi,hi,:))-Dce_term)./r2_p;
            else
                Ct_corr_t2(wi,hi,:) = squeeze(data_R2(wi,hi,:))./r2_p;
            end
        end
    end
end

%% Cal CBV CBF
type = 'cSVD';

% Ct from leakage-corrected T2/T2star
[CBV_t2star_corr]=DSC_mri_cbv(Ct_corr_t2star.*mask,AIF,mask,TR,Hf,rho);
[CBF_t2star_corr]=DSC_mri_cbf_leakage(Ct_corr_t2star.*mask,AIF,mask,TR,Hf,rho,type,options);

[CBV_t2_corr]=DSC_mri_cbv(Ct_corr_t2.*mask,AIF,mask,TR,Hf,rho);
[CBF_t2_corr]=DSC_mri_cbf_leakage(Ct_corr_t2.*mask,AIF,mask,TR,Hf,rho,type,options);

% Ct from leakage-corrupted T2/T2star
[Ct_t2star,~] = caculate_Ct(data_t2star,mask,nSamplesMax,r2star_p);
[CBV_t2star]=DSC_mri_cbv(Ct_t2star,AIF,mask,TR,Hf,rho);
[CBF_t2star]=DSC_mri_cbf_leakage(Ct_t2star,AIF,mask,TR,Hf,rho,type,options);

[Ct_t2,~] = caculate_Ct(data_t2,mask,nSamplesMax,r2_p);
[CBV_t2]=DSC_mri_cbv(Ct_t2,AIF,mask,TR,Hf,rho);
[CBF_t2]=DSC_mri_cbf_leakage(Ct_t2,AIF,mask,TR,Hf,rho,type,options);

%% show results
figure(111);
subplot(2,2,1);imshow(CBV_t2star,[0 10]);title('CBV T2star');colorbar;
subplot(2,2,2);imshow(CBF_t2star,[0 100]);title('CBF T2star');colorbar;
subplot(2,2,3);imshow(CBV_t2star_corr,[0 10]);title('CBV T2star Corr');colorbar;
subplot(2,2,4);imshow(CBF_t2star_corr,[0 100]);title('CBF T2star Corr');colorbar;
colormap(hotIronMap);

figure(222);
subplot(2,2,1);imshow(CBV_t2,[0 10]);title('CBV T2');colorbar;
subplot(2,2,2);imshow(CBF_t2,[0 100]);title('CBF T2');colorbar;
subplot(2,2,3);imshow(CBV_t2_corr,[0 10]);title('CBV T2 Corr');colorbar;
subplot(2,2,4);imshow(CBF_t2_corr,[0 100]);title('CBF T2 Corr');colorbar;
colormap(hotIronMap);