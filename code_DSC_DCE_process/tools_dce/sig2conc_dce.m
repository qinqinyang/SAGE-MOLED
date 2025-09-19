function C = sig2conc_dce(img,R10,alpha,TR,imgB,mask)
% functon to calculate contrast concentration from image intensity
% equation: 
% R1(t)=-1/TR*ln(1-((S(t)-S(0))/S0*sin(alpha))+(1-m)/(1-m*cos(alpha)))
%over 1-cos(alpha)*((S(t)-S(0))/S0*sin(alpha))+(1-m)/(1-m*sin(alpha)))
% where m=exp(-R10*TR)
% then C(t)=(R1(t)-R1(0))/r1

% Yi Guo, 06/12/2014
% No mask comparing to Marc's version, otherwise allmost the same
% some simplification, pay attention to R1, and alpha unit!!!

Rcs=3.47; % contrast relaxivity
% nt=size(img,3); % temporal dimension size

A = R10 .* TR;
B = Rcs * TR;
m = exp(-A);

par1 = (img.*(1-m))./(imgB.*(1-cos(alpha).*m))-1;
par2 = (img.*cos(alpha).*(1-m))./(imgB.*(1-cos(alpha).*m))-1;

par3 = A + log(par1./par2);

C = -par3./B;

% m=exp(-R10.*TR);
% M0 = (imgB.*(1-m.*cos(alpha)))./(sin(alpha).*(1-m));
% 
% par1 = repmat(M0,[1 1 nt]).*sin(alpha)-img;
% par2 = repmat(M0,[1 1 nt]) - img.*cos(alpha);
% B = par1./par2;

% par2=(1-m)./(1-m.*cos(alpha));
% 
% par1=(img-repmat(imgB,[1 1 nt]))./repmat(M0+eps,[1 1 nt])./sin(alpha);
% 
% B=(par1+par2);
% 
% Rt=-1./TR*real(log((1-B)./(1-cos(alpha)*B+eps)));

% R1B=R10; % baseline R1 is equal to R10 if img(1)==imgB
% R1B=R10;

% R1 = -1./TR.*log(B);
% C=R1-repmat(R10,[1 1 nt]);

% C=C./Rcs;

C(C<0) = 0;  % get rid of outliers
C(C>180)=0;
C=abs(C.*mask);
end

    
    



