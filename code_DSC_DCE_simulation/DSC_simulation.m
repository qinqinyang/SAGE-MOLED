clear,clc;
load("Simu_Template.mat");
load("Simu_AIF.mat");
rho = 1.04;
Hf = 0.7;

TR = 1.9;
time_in=19;
tpres=TR/60; % temporal resolution, unit in seconds!
frames = size(ca,2);
time=[zeros(1,time_in),[1:(frames-time_in)]*TR];
time = time(time_in:end);
frames_r = size(time,2);

%% R(t)
[ww,hh] = size(CBF);
R = zeros(ww,hh,frames_r);
for wi = 1:ww
    for hi = 1:hh
        if CBV(wi,hi) > 0
            cbf_temp = CBF(wi,hi)./(60);
            cbv_temp = CBV(wi,hi);
            R(wi,hi,:) = exp(-cbf_temp.*time./cbv_temp);
        end
    end
end

%% Ct(t)
Ct = zeros(ww,hh,frames);
AIF = ca;
for wi = 1:ww
    for hi = 1:hh
        if CBV(wi,hi) > 0
            cbf_temp = CBF(wi,hi)./(100.*60);
            R_temp = R(wi,hi,:);
            R_temp = R_temp(:);
            Ct(wi,hi,:) = rho./Hf.*cbf_temp.*filter(R_temp,1,AIF);
        end
    end
end

%%
figure;
imshow3(Ct,[0,0.6]);colormap jet;

%%
figure;
R_show = R(256,256,:);
plot(R_show(:));
figure;
plot(AIF(:));

