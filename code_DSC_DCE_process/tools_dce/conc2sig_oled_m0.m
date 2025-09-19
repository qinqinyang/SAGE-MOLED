function Mz = conc2sig_oled_m0(m0,TR,R1_dyn)

apha=pi/6;
E = exp(-TR.*R1_dyn);
A1 = m0.*(1-E);
A0 = 1+E.*cos(apha)^4;

Mz = A1./A0;
Mz(isnan(Mz)) = 0;
Mz(Mz<0) = 0;  % get rid of outliers
end

    
    



