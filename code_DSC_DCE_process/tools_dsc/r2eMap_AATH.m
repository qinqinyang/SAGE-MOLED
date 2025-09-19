function [r2p_data, r2e_data] = r2eMap_AATH(Delta_R2, Ktrans, ve, Tc, Fp, time, Cp)
[w,h,f]=size(Delta_R2);

r2e_data=zeros(w,h);
r2p_data=zeros(w,h);

x0=[10 10];
lb=[0 0];
ub=[20000 150];

options = optimset('Display','off');

parfor wi=1:w
    for hi=1:h

        temp = squeeze(Delta_R2(wi,hi,:));
        ve_temp = ve(wi,hi);
        Ktrans_temp = Ktrans(wi,hi);
        Tc_temp = Tc(wi,hi);
        Fp_temp = Fp(wi,hi);

        if  ve_temp > 0.01 && Ktrans_temp > 0.01 

            model = @(r2,time) model_tissue_homogeneity_dsc(r2, time, Ktrans_temp, ve_temp, Fp_temp, Tc_temp, Cp);
            r2 = lsqcurvefit(model,x0,time,temp,lb,ub,options);

            r2p_data(wi,hi) = r2(1);			% ktrans
            r2e_data(wi,hi) = r2(2);

        end
    end
    disp(['Finish line ',num2str(wi)]);
end

end