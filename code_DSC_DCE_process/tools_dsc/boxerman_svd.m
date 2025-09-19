function Ct_corr = boxerman_svd(Ct,C_ref)

% get boxerman constants K1 and K2 from SVD
 % C_ref = referenece (non-leaky) response curve 
 % Ct = leaky response curve
 % Ct_corr = leakage corrected response curve
 % N = # of datapnts

N = length(C_ref);

 CC=zeros(N,2);
 CC(:,1)=C_ref';
 for i=1:N,
     CC(i,2)=sum(C_ref(1:i));
 end
 %do SVD deconv
 [U,S,V]=svd(CC);
 L=diag(S);
 b=zeros(2,1);
 for j=1:2,
     UU=1/L(j)*(U(:,j)'*Ct);
     b=b+UU*V(:,j);
 end
 % estimate leakage corrected response curve
 for i=1:N,
     Ct_corr(i)=Ct(i)-b(2)*CC(i,2);
 end
  % b(1) = blood volume fraction (C_ref vs Ct) = K1 (Boxerman)
 % b(2) = leakage constant =K2 (Boxerman)
 