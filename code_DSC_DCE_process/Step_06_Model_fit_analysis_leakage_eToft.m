clc,clear,close all;
fn = 'F:\我的坚果云\论文-SAGE-MOLED\data_paper\Sub23_胶质瘤\DCE_eToft_AIF_T1_real_for_analysis_analysis.mat';
load(fn);

figure;
xx = 128;
yy = 158;
data_R2star = data_R2;
temp = squeeze(data_R2star(xx,yy,:));
temp_ktrans = ktrans_mask(xx,yy);
temp_ve = ve_mask(xx,yy);
temp_vp = vp_mask(xx,yy);
plot(temp);
figure;
plot(AIF);
Nt = size(data_R2star,3);

%% produce a fake image
data_R2star = zeros(4,4,Nt);
ktrans_mask = zeros(4,4);
ve_mask = zeros(4,4);
vp_mask=zeros(4,4);

for i=1:4
    for j=1:4
        data_R2star(i,j,:)=temp;
        ktrans_mask(i,j) = temp_ktrans;
        ve_mask(i,j) = temp_ve;
        vp_mask(i,j) = temp_vp;
    end
end

maxValue = max(AIF);
index = find(AIF == maxValue);
time_in=index-1;
tpres=1.9/60; % temporal resolution, unit in seconds!
% time=[zeros(1,time_in),[1:(Nt-time_in)]*tpres];
time=[zeros(1,time_in),[1:(Nt-time_in)]*tpres];
[r2p_data, r2e_data] = r2eMap_ET(data_R2star, ktrans_mask, ve_mask, vp_mask, time, AIF);
r2p_data = r2p_data(1,1)
r2e_data = r2e_data(1,1)

%
ridex(1)=r2p_data;
ridex(2)=r2e_data;
Ct_temp = model_extended_tofts_dsc(ridex,time, temp_ktrans, temp_ve, temp_vp,AIF);
Ct_ex_temp = model_extended_tofts_dsc_ex(ridex(2),temp_ktrans, temp_ve, AIF, time);
Ct_in_temp = model_extended_tofts_dsc_in(ridex(1),temp_vp, AIF, time);

figure(111);
subplot(311);plot(temp);hold on,plot(Ct_temp);grid on;title('Total model');
subplot(312);plot(temp);hold on,plot(temp-Ct_ex_temp);grid on;title('Ex model');
subplot(313);plot(temp);hold on,plot(Ct_in_temp);grid on;title('In model');