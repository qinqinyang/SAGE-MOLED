function C = sig2conc_oled_v1(img,R10,TR,S0,mask)

Rcs=3.6; % contrast relaxivity

nt=size(img,3); % temporal dimension size

e=1-exp(-repmat(R10,[1 1 nt]).*TR);
S0=abs(S0)*1000;
st=abs(img)*1000;
stt=((st.*e)./S0);

R1t=-1/TR.*log(1-stt);

C=abs(R1t)-repmat(R10,[1 1 nt]);
C=C./Rcs;

C(C<0) = 0;  % get rid of outliers
C=abs(C.*mask);
end

    
    



