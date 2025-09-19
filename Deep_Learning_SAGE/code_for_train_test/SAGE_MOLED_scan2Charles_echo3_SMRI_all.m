% Make Charles for network training
% Date: 2022-10-24
% User: Qinqin Yang
% --------------------------

clear;clc;

srcUrl = 'scan_paper/scan_1954_rand_noB0/';
fieldUrl = 'scan_paper/scan_1954_rand_noB0_field/';
vobjUrl = 'Template_SMRI_rand/';
dstUrl = 'scan_paper/train_rand/';
topname='rand_';

mkdir(dstUrl);

FRE_NUM = 128;
PHASE_NUM = 128;
h=128;
w=128;

EXPAND_NUM = 256;
gy=2.67519e8;
pi=3.14159265359;

start_index = 1;
end_index = 5000;

for index = start_index:end_index
    
    srcName = sprintf("%d.out", index);
    fieldName = sprintf("%d.mac", index);
    vobjName = sprintf("%d.mat", index);
    
    srcPath = fullfile(srcUrl, srcName);
    fieldPath = fullfile(fieldUrl, fieldName);
    vobjPath = fullfile(vobjUrl, vobjName);
    
    load(vobjPath);
    
    current_out = zeros(3, EXPAND_NUM, EXPAND_NUM);
    fielddata = SMri2D_reader(fieldPath, 500, 500);
    fielddata = permute(fielddata,[2,3,1]);
    
    T2 = VObj.T2;
    T2 = abs(imresize(T2,[EXPAND_NUM,EXPAND_NUM],'nearest'));
    T2 = flip(rot90(T2,3),1);
    
    T2star = VObj.T2Star;
    T2star = abs(imresize(T2star,[EXPAND_NUM,EXPAND_NUM],'nearest'));
    T2star = flip(rot90(T2star,3),1);
    
    M0 = VObj.Rho;
    M0 = abs(imresize(M0,[EXPAND_NUM,EXPAND_NUM],'nearest'));
    M0 = flip(rot90(M0,3),1);
    mask = zeros(EXPAND_NUM,EXPAND_NUM);
    mask(M0>0)=1;
    
    B1 = fielddata(:,:,1);
    B1 = imresize(B1,[EXPAND_NUM,EXPAND_NUM],'nearest');
    B1 = flip(rot90(B1,3),1);
     
    B0 = fielddata(:,:,2);
    B0 = imresize(B0,[EXPAND_NUM,EXPAND_NUM],'nearest');
    B0 = flip(rot90(B0,3),1);
    B0 = B0*gy/2/pi;
     
    alldata = SMri2D_reader(srcPath, h, w);
    
    % echo1
    k1_real_data = reshape(alldata(1,:,:,1), [h,w]);
    k1_imag_data = reshape(alldata(2,:,:,1), [h,w]);
    k1_ori_kspace = k1_real_data + 1i*k1_imag_data;
    k1_ori_kspace(:,2:2:end) = flipud(k1_ori_kspace(:,2:2:end));
    ksp_echo1 = rot90(k1_ori_kspace(:,1:PHASE_NUM),2);
    
    % echo2
    k2_real_data = reshape(alldata(1,:,:,2), [h,w]);
    k2_imag_data = reshape(alldata(2,:,:,2), [h,w]);
    k2_ori_kspace = k2_real_data + 1i*k2_imag_data;
    k2_ori_kspace(:,2:2:end) = flipud(k2_ori_kspace(:,2:2:end));
    ksp_echo2 = flip(k2_ori_kspace(:,1:PHASE_NUM),2);
    
    % echo3
    k3_real_data = reshape(alldata(1,:,:,3), [h,w]);
    k3_imag_data = reshape(alldata(2,:,:,3), [h,w]);
    k3_ori_kspace = k3_real_data + 1i*k3_imag_data;
    k3_ori_kspace(:,2:2:end) = flipud(k3_ori_kspace(:,2:2:end));
    ksp_echo3 = rot90(k3_ori_kspace(:,1:PHASE_NUM),4);
    
    % add noise in kspace
    ksp_echo_img1 = ifft2c(ksp_echo1);
    ksp_echo_img2 = ifft2c(ksp_echo2);
    ksp_echo_img3 = ifft2c(ksp_echo3);
    
    [rand_map1,rand_small_map]=creat_echo_train_gap(h,1.0,0.3);
    cup = 600;
    
    ksp_echo_img1 = ksp_echo_img1 ./ cup;
    ksp_echo_img2 = ksp_echo_img2 ./ cup .*rand_map1;
    ksp_echo_img3 = ksp_echo_img3 ./ cup .*rand_map1;
    
    rand_factor=0.14*rand();
    k1_noise = rand_factor*(randn(FRE_NUM,PHASE_NUM))+rand_factor*1.0i*(randn(FRE_NUM,PHASE_NUM));
    k2_noise = rand_factor*(randn(FRE_NUM,PHASE_NUM))+rand_factor*1.0i*(randn(FRE_NUM,PHASE_NUM));
    k3_noise = rand_factor*(randn(FRE_NUM,PHASE_NUM))+rand_factor*1.0i*(randn(FRE_NUM,PHASE_NUM));
    ksp_img_noise1 = ksp_echo_img1 + k1_noise;
    ksp_img_noise2 = ksp_echo_img2 + k2_noise;
    ksp_img_noise3 = ksp_echo_img3 + k3_noise;
    ksp_echo1 = fft2c(ksp_img_noise1);
    ksp_echo2 = fft2c(ksp_img_noise2);
    ksp_echo3 = fft2c(ksp_img_noise3);
    
    k_expand = zeros(EXPAND_NUM, EXPAND_NUM,3) + 1.0i * zeros(EXPAND_NUM, EXPAND_NUM,3);
    k_expand(round((EXPAND_NUM-FRE_NUM)/2)+1:round((EXPAND_NUM+FRE_NUM)/2),round((EXPAND_NUM-PHASE_NUM)/2)+1:round((EXPAND_NUM+PHASE_NUM)/2),1)=ksp_echo1;
    k_expand(round((EXPAND_NUM-FRE_NUM)/2)+1:round((EXPAND_NUM+FRE_NUM)/2),round((EXPAND_NUM-PHASE_NUM)/2)+1:round((EXPAND_NUM+PHASE_NUM)/2),2)=ksp_echo2;
    k_expand(round((EXPAND_NUM-FRE_NUM)/2)+1:round((EXPAND_NUM+FRE_NUM)/2),round((EXPAND_NUM-PHASE_NUM)/2)+1:round((EXPAND_NUM+PHASE_NUM)/2),3)=ksp_echo3;
    k_image=ifft2c(k_expand);
    rand_map1 = imresize(rand_map1,[256,256]);
    
    %k_max3norm = max(abs(k_image(:)))
    %k_image = k_image/0.5;
    
    current_out(1,:,:) = real(k_image(:,:,1));
    current_out(2,:,:) = imag(k_image(:,:,1));
    current_out(3,:,:) = real(k_image(:,:,2));
    current_out(4,:,:) = imag(k_image(:,:,2));
    current_out(5,:,:) = real(k_image(:,:,3));
    current_out(6,:,:) = imag(k_image(:,:,3));
    
    current_out(7,:,:) = T2;
    current_out(8,:,:) = T2star;
    current_out(9,:,:) = M0;
    current_out(10,:,:) = B0.*mask;
    current_out(11,:,:) = B1.*mask;
    current_out(12,:,:) = rand_map1.*mask;
    
    if sum(isnan(current_out(:)))==0
        % save file to Charles
        save_name = [topname,num2str(index,'%04d'),'.Charles'];
        save_path = fullfile(dstUrl, save_name);
        
        [fid,msg]=fopen(save_path, 'wb');
        fwrite(fid, current_out, 'single');
        fclose(fid);
        disp(save_path);
    else
        disp('error');
    end
end