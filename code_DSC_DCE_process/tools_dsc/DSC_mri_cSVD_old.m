function map=DSC_mri_cSVD_old(conc,aif,mask,TR)

threshold1=0.05;

[nR,nP,nT] = size(conc);

% 1) Creo la matrice G
nTpad=2*nT;
columnG=zeros(nTpad,1);
columnG(1)=aif(1);
columnG(nT)=(aif(nT-1)+4*aif(nT))/6;
columnG(nT+1)=aif(nT)/6;
for k=2:(nT-1)
    columnG(k)=(aif(k-1)+4*aif(k)+aif(k+1))/6;
end
rowG=zeros(1,nTpad);
rowG(1)=columnG(1);
for k=2:nTpad
    rowG(k)=columnG(nTpad+2-k);
end

G=toeplitz(columnG,rowG);

[U,S,V]=svd(G);

eigenV=diag(S);
threshold=threshold1*max(eigenV);    % threshold del 10% con in Ostergaard e Calamante
newEigen=zeros(size(eigenV));
for k=1:length(eigenV)
    if eigenV(k)>=threshold
        newEigen(k)=1/eigenV(k);
    end
end

Ginv=V*diag(newEigen)*(U');

map=zeros(nR,nP);

for r=1:nR
    for c=1:nP
            if mask(r,c)
                % Calcolo la funzione residuo
                vettConc=zeros(nTpad,1);
                vettConc(1:nT)=reshape(conc(r,c,:),nT,1);
                vettRes=(1/TR)*Ginv*vettConc;
                
                map(r,c)=max(abs(vettRes));
            end
    end
end
end