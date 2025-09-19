%% DCE simulation eToft Model
clear,clc;
load("Simu_Template.mat");
load("Simu_AIF.mat")

%%
TR = 1.9;
time_in=19;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(ca,2);
time=[zeros(1,time_in),[1:(frames-time_in)]*tpres];
Cp = ca';

Ct = zeros(512,512,frames);
parfor ii=1:512
    for jj=1:512
        if M0(ii,jj) > 0
            Ct(ii,jj,:) = model_extended_tofts_simu(ktrans(ii,jj), ve(ii,jj), vp(ii,jj), Cp, time);
        end
    end
    disp(ii);
end

%%
figure;
conf_ref = imresize(Ct,[256,256]);
imshow3(conf_ref,[0 0.5]);colormap jet;
