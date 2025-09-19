function S0map=DSC_cal_S0_mean(volumes,mask,nSamplesMax)

S0map=mask.*mean(volumes(:,:,3:nSamplesMax),3);

end