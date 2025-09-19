clear,clc;

workpath = 'F:\我的坚果云\论文-SAGE-MOLED\data_paper\Sub07_脑膜瘤\meas_MID02719_FID1377060_a_sage_oled_1954_IPAT2_Dynamic_SENSE_Charles_T2T2star\';
fn=[workpath,'slice_007.mat'];
fn_AIF = 'F:\我的坚果云\论文-SAGE-MOLED\data_paper\Sub07_脑膜瘤\AIF_filter.mat';

load(fn); % [256   256   110     2]
load(fn_AIF);
AIF = AIF_fit';
maxValue = max(AIF);
index = find(AIF == maxValue);
nSamplesMax = index-5; % time point of CA injection

GE_image = results(:,:,:,2);
GE_image = rev_tensor(GE_image);
GE_image = rot90(GE_image,2);
[Nx,Ny,Nt] = size(GE_image);

% for z = 6
GE_slice = squeeze(GE_image);

GE_total = zeros(Nx,Ny,Nt);

for t = 1:Nt
    GE_total(:,:,t) = GE_slice(:,:,t);
end

GE_mean_baseline = mean(GE_image(:,:,1:nSamplesMax),3);

ref_image_GRE_slice = squeeze(GE_mean_baseline);

% Mask generation from T2
data_r = results(:,:,:,1);
data_r = rev_tensor(data_r);
data_t2 = rot90(data_r,2);

SE_mean_baseline = mean(data_t2(:,:,1:nSamplesMax),3);
T = 0.1;
brain_mask = mask_generate(SE_mean_baseline,T);

figure;imshow([ref_image_GRE_slice,brain_mask,ref_image_GRE_slice.*brain_mask],[0 0.15]);
figure;
imshow3(GE_total,[0 0.2]),colormap jet;
%%
% cal GE mean value
mean_GRE_slice_value = zeros(Nt,1);

for t = 1:Nt
    mean_GRE_slice_value(t) = mean(nonzeros(GE_total(:,:,t).*brain_mask));
end

smooth_mean_GRE =smooth(mean_GRE_slice_value(1:end),'moving');
R2_mean_GRE = caculate_R2_single(smooth_mean_GRE,nSamplesMax);

Ct_t2star = zeros(Nx,Ny,Nt);

for x = 1:Nx
    for y = 1:Ny
        if  brain_mask(x,y)
            R2_GRE_raw = squeeze(GE_total(x,y,:));
            smooth_GRE = smooth(R2_GRE_raw(1:end),'moving');
            R2_GRE = caculate_R2_single(smooth_GRE,nSamplesMax);
            Ct_t2star(x,y,:) = (boxerman_svd(R2_GRE,R2_mean_GRE))';
        end
    end
end

%%
figure;
imshow3(Ct_t2star,[0 30]);colormap jet;

rho = 1.04;
Hf = 0.7;

TR = 1.9;
r2s = 87;      % uint: L/mmol/s T2star
r2 = 20.4;     % uint: L/mmol/s T2

type = 'cSVD';
options.display = 1;
options.nT = Nt;
options.deconv.cSVD.threshold = 0.01;
options.deconv.cSVD.residual = 1;
options.deconv.SVD.residual = 1;
options.nR = 256;
options.nC = 256;
options.waitbar = 1;
options.tr = 1.9;

[CBV_t2star]=DSC_mri_cbv(Ct_t2star,AIF,brain_mask,TR,Hf,rho);
[CBF_t2star]=DSC_mri_cbf_leakage(Ct_t2star,AIF,brain_mask,TR,Hf,rho,type,options);

% cal CBV CBF
hotIronColors = [
    0, 0, 0;     % 黑色
    0, 0.5, 1;   % 青色
    0, 0.6, 0;   % 绿色
    1, 1, 0;     % 黄色
    1, 0, 0;     % 红色
    1, 1, 1;     % 白色
];

hotIronMap = interp1(linspace(0, 1, size(hotIronColors, 1)), hotIronColors, linspace(0, 1, 256));

figure(222);
subplot(1,2,1);imshow(CBV_t2star,[0 1000]);title('CBV T2star');colorbar;
subplot(1,2,2);imshow(CBF_t2star,[0 15000]);title('CBF T2star');colorbar;
colormap(hotIronMap);