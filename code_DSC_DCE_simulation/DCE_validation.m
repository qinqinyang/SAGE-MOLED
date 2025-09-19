%% DCE simulation eToft Model
clear,clc;
load("Simu_Template.mat");
load("Simu_AIF.mat")

%% resize 
ktrans_temp = imresize(ktrans,[128,128],'nearest');
ve_temp = imresize(ve,[128,128],'nearest');
vp_temp = imresize(vp,[128,128],'nearest');
M0_temp = imresize(M0,[128,128],"nearest");

%%
TR = 1.9;
time_in=19;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(ca,2);
time=[zeros(1,time_in),[1:(frames-time_in)]*tpres];
Cp = ca';

Ct = zeros(128,128,frames);
parfor ii=1:128
    for jj=1:128
        if M0_temp(ii,jj) > 0
            Ct(ii,jj,:) = model_extended_tofts_simu(ktrans_temp(ii,jj), ve_temp(ii,jj), vp_temp(ii,jj), Cp, time);
        end
    end
    disp(ii);
end
figure;
imshow3(Ct,[0 0.5]);colormap jet;

%%
figure;
imshow3(Ct(:,:,1:15:100),[0 0.4],[1,7]);colormap jet;

%%
mask=zeros(size(M0_temp));
mask(M0_temp>0)=1;
[ktrans_temp2,ve_temp2,vp_temp2]=conc2Ktrans_eTofts(Ct,time,Cp,mask);

%%
figure(11);
subplot(1,3,1);imshow(ktrans_temp,[0 0.3]);title('Kt','FontSize',32);colorbar;
subplot(1,3,2);imshow(vp_temp,[0 0.04]);title('Vp','FontSize',32);colorbar;
subplot(1,3,3);imshow(ve_temp,[0 0.4]);title('Ve','FontSize',32);colorbar;
colormap jet;

figure(12);
subplot(1,3,1);imshow(ktrans_temp2,[0 0.3]);title('Kt','FontSize',32);colorbar;
subplot(1,3,2);imshow(vp_temp2,[0 0.04]);title('Vp','FontSize',32);colorbar;
subplot(1,3,3);imshow(ve_temp2,[0 0.4]);title('Ve','FontSize',32);colorbar;
colormap jet;

figure(13);
subplot(1,3,1);imshow(abs(ktrans_temp2-ktrans_temp)*10,[0 0.3]);title('Kt','FontSize',32);colorbar;
subplot(1,3,2);imshow(abs(vp_temp2-vp_temp)*10,[0 0.04]);title('Vp','FontSize',32);colorbar;
subplot(1,3,3);imshow(abs(ve_temp2-ve_temp)*10,[0 0.4]);title('Ve','FontSize',32);colorbar;
colormap jet;
