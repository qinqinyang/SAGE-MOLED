clear,clc;
load("Simu_Template.mat");
load("Simu_AIF.mat");
rho = 1.04;
Hf = 0.7;

TR = 1;
frames = size(ca,2);

time = 0:TR:frames*TR-1;
t=0:frames-1;
%% R(t) and Ct(t)
[ww,hh] = size(CBF);
R = zeros(ww,hh,frames);
Ct = zeros(ww,hh,frames);
P = zeros(ww,hh);
AIF = ca;
for wi = 1:ww
    for hi = 1:hh
        if CBV(wi,hi) > 0
            cbf_temp = CBF(wi,hi)./(100.*60);
            cbv_temp = CBV(wi,hi)./(100);
            R_temp = exp(-cbf_temp.*t./cbv_temp);
            R_temp = R_temp(:);
            Ct(wi,hi,:) = rho./Hf.*cbf_temp.*filter(R_temp,1,AIF);
            P(wi,hi)= trapz(Ct(wi,hi,:));
        end
    end
end

Q=trapz(t,AIF);
CBV = 100 *(Hf./rho)* P./Q;

figure;
Ct_show = Ct(256,256,:);
plot(Ct_show(:));
%%
figure;
imshow3(Ct,[0,0.6]);colormap jet;

%%
figure;
slist = [1:5:110-6];
imshow3(Ct(:,:,slist),[0 0.5],[3,7]);colormap jet;

%% CBV CBF
type = 'cSVD';

[CBV2]=DSC_mri_cbv(Ct,AIF,mask,TR,Hf,rho);
[CBF2]=DSC_mri_cbf(Ct,AIF,mask,TR,Hf,rho,type);

%%
figure;
subplot(1,2,1);imshow(CBV,[0 20]);title('CBV');colorbar;
subplot(1,2,2);imshow(CBF,[0 150]);title('CBF');colorbar;
colormap jet;

figure;
subplot(1,2,1);imshow(CBV2,[0 20]);title('CBV');colorbar;
subplot(1,2,2);imshow(CBF2,[0 150]);title('CBF');colorbar;
colormap jet;

figure;
subplot(1,2,1);imshow(CBV./CBV2,[0 5]);title('CBV');colorbar;
subplot(1,2,2);imshow(CBF./CBF2,[0 5]);title('CBF');colorbar;
colormap jet;