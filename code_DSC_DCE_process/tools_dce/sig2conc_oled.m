function C = sig2conc_oled(img,R10,TR,S0map,mask)

Rcs=3.6; % contrast relaxivity
apha=pi/6;
nt=size(img,3); % temporal dimension size

M0 = abs(S0map);
Mt = abs(img);
E0 = Mt./repmat(M0,[1 1 nt]);
E1 = exp(-TR.*repmat(R10,[1 1 nt]));
E = (1+cos(apha)^4.*E1)./(1-E1);
R1t = (E + E0*cos(apha)^4)./(E-E0);
R1t = log(R1t)./TR;

C=(R1t-R10)./Rcs;

C(C<0) = 0;  % get rid of outliers
C=abs(C.*mask);
end

    
    



