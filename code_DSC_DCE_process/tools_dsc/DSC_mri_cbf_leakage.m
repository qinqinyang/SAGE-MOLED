function [cbf]=DSC_mri_cbf_leakage(conc,aif,mask,TR,kh,rho,type,options)
switch type
    case 'SVD'
        cbf=DSC_mri_cSVD(conc,aif,mask,TR);
    case 'cSVD'
        cbf=DSC_mri_cSVD(conc,aif,mask,options);
end
cbf = 100*60*(kh/rho)*cbf.map;
cbf(isnan(cbf)==1) = 0;
cbf=abs(cbf);
cbf(cbf==inf)=0;
cbf(cbf<0)=0;
%cbf(cbf>200)=200;
end