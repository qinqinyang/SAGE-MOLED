function vettRes=Deconv_cSVD_tv(conc,aif,mask,lambda)

[nR,nP,nT] = size(conc);

Msize = nT;
G = zeros(Msize,Msize);
rowG = zeros(1,nT);
rowG(1,1:nT) = aif;
for Mi = 1:Msize
    G(:,Mi)=rowG;
    rowG = circshift(rowG,1);
end

G = tril(G);

[U,S,V]=svd(G,'econ');

eigenV=diag(S);

S_reg = diag(eigenV ./ (eigenV.^2 + lambda^2));
Ginv=V*S_reg*(U');
vettRes=zeros(nR,nP,nT);

for r=1:nR
    for c=1:nP
        if mask(r,c)
            vettConc=squeeze(conc(r,c,:));
            tempRes = Ginv*vettConc;
            vettRes(r,c,:)= tempRes;
        end
    end
end
end