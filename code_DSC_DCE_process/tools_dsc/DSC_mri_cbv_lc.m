function [cbv_corrected,K1_map,K2_map,K1_CV_map,K2_CV_map]=DSC_mri_cbv_lc(conc,aif,mask,bolus,options)

cbv=zeros(options.nR,options.nC);

SD = vol2mat (conc,logical(mask));
SD = nanstd(SD(:,1:floor(mean(bolus(:)))),[],2);
SD_map = zeros(size(cbv));
SD_map(logical(logical(mask))) = SD;

AVG = vol2mat (conc,logical(mask));
AVG = nanmedian(AVG(:,end-10:end),2);
AVG_map = zeros(size(cbv));
AVG_map(logical(mask)) = AVG;


mask_not_enhancing = and(logical(mask), not(abs(AVG_map) > 2.*SD_map));
conc(isinf(conc)) = 0;
R2star_AVG_not_enhancing = nanmean(vol2mat(conc,mask_not_enhancing))';

Delta_R2star = vol2mat(conc,logical(mask));

phat = zeros(size(Delta_R2star,1),2);
CVp = zeros(size(Delta_R2star,1),2);
A = [-cumtrapz(options.time,R2star_AVG_not_enhancing)  R2star_AVG_not_enhancing];
sigmaphat = inv(A'*A);

bolus_min = min(bolus(:));

for v=1:size(Delta_R2star,1)
    Delta_R2star_vett = Delta_R2star(v,:)';

    sigma2 = (std(Delta_R2star_vett(1:bolus_min))).^2;
    
    B = Delta_R2star_vett;
    phat(v,:) = A\B;
    temp = sigma2.*sigmaphat;
    CVp(v,:) = 100.*sqrt([temp(1,1) temp(2,2)])./abs(phat(v,:));
end

K2_vett  = phat(:,1);
K1_vett  = phat(:,2);

K2_CV_vett  = CVp(:,1);
K1_CV_vett  = CVp(:,2);

K2_map  = zeros(size(cbv));
K1_map  = zeros(size(cbv));

K2_map(logical(mask)) = K2_vett;
K1_map(logical(mask)) = K1_vett;

K2_CV_map = zeros(size(cbv));
K1_CV_map = zeros(size(cbv));


K2_CV_map(logical(mask)) = K2_CV_vett;
K1_CV_map(logical(mask)) = K1_CV_vett;

cbv=mask.* ...
        (trapz(options.time,conc,3));

cbv_corrected = ((cbv + (K2_CV_map<100).*K2_map.*trapz(options.time,cumtrapz(options.time,R2star_AVG_not_enhancing)))./trapz(options.time,aif));

end
