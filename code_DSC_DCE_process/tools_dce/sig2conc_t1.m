function C = sig2conc_t1(img,R10,mask)

Rcs=3.6; % contrast relaxivity

R1t = (1./img);
R1t(isnan(R1t)==1)=0;
C=(R1t-R10)./Rcs;

C(C<0) = 0;  % get rid of outliers
C=abs(C.*mask);
end

    
    



