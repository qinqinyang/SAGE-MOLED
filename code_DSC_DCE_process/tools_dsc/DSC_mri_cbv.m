function [cbv]=DSC_mri_cbv(conc,aif,mask,TR,kh,rho)

[nR,nP,nT]  = size(conc);
times = [0:TR:nT*TR-1];

mask = double(mask);

cbv=100*(kh/rho).*mask.* ...
    (trapz(times,conc,3)./trapz(times,aif));

end
