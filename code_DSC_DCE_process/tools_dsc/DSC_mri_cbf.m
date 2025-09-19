function [cbf]=DSC_mri_cbf(conc,aif,mask,TR,kh,rho,type)
switch type
    case 'SVD'
        cbf=DSC_mri_cSVD(conc,aif,mask,TR);
    case 'cSVD'
        cbf=DSC_mri_cSVD(conc,aif,mask,TR);
end
cbf = 100*60*(kh/rho)*cbf;
cbf(isnan(cbf)==1) = 0;
cbf=abs(cbf);
cbf(cbf==inf)=0;
cbf(cbf<0)=0;
%cbf(cbf>200)=200;
end