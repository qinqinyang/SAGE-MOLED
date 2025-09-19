function [mtt]=DSC_mri_mtt(cbv,cbf)

mtt = 60*cbv./cbf;

mtt=abs(mtt);
mtt(isnan(mtt)==1) = 0;
mtt(mtt==inf)=0;