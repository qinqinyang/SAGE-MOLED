function [CBV,CBF,MTT,TTP] = get_CBV_CBF_MTT(Ct,Ca,type)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[nR,nP,nT] = size(Ct);

Q= trapz(Ca);
P = zeros(nR,nP);
y1 = Ca;
CBF=zeros(nR,nP);
switch type
    case 'SVD'
        for i=2:nT-1
            y1(i) =   ( y1(i-1) + 4*y1(i) + y1(i+1) )/6;
        end

        A=toeplitz(y1,[y1(1),zeros(1,nT+9)]);
        [U,S,V]=svd(A);
        diag_value=diag(S);
        threshord=0.2*max(diag_value);
        diag_value(diag_value<threshord)=0;
        diag_value=diag_value.^-1;
        diag_value(isinf(diag_value))=0;
        G=V*diag(diag_value)*(U');

        for j=1:nR
            for i=1:nP
                P1 = squeeze (Ct(i,j,:));
                P(i,j)= trapz(P1);
                CBF(i,j)=max(G*P1);
            end
        end
    case 'cSVD'

        columnA=zeros(nT-1+100,1);
        columnA(1)=y1(1);
        columnA(nT)=(y1(nT-1)+4*y1(nT-1))/6;
        columnA(nT+1)=y1(nT)/6;

        for k=2:nT-1
            columnA(k)=(y1(k-1)+4*y1(k)+y1(k+1))/6;
        end
        rowA=zeros(1,nT-1+100);
        rowA(1)=columnA(1);
        for k=2:nT-1+100
            rowA(k)=columnA(nT-1+100+2-k);
        end
        A2=toeplitz(columnA,rowA);
        [U,S,V]=svd(A2);

        eigenV=diag(S);
        threshold=0.1*max(eigenV);    % Threshold: 10%
        newEigen=zeros(size(eigenV));
        for k=1:length(eigenV)
            if eigenV(k)>=threshold
                newEigen(k)=1/eigenV(k);
            end
        end

        G2=V*diag(newEigen)*(U');



        for r=1:nR
            for c=1:nP
                P1=squeeze(Ct(r,c,:));
                P(r,c)= trapz(P1);

                vettConc=zeros(nT-1+100,1);
                vettConc(1:nT)=Ct(r,c,:);
                CBF(r,c)=max(abs(G2*vettConc));
            end
        end
end

CBV = P./Q ;
CBV(isnan(CBV)==1) = 0;
CBV(CBV==inf)=0;
CBV(CBV<0)=0;
line_cbv=sort(CBV(:),'DESCEND');
CBV(CBV>line_cbv(20))=line_cbv(20);

CBF(isnan(CBF)==1) = 0;
CBF=abs(CBF);
CBF(CBF==inf)=0;
CBF(CBF<0)=0;
line_cbf=sort(abs(CBF(:)),'DESCEND');
CBF(CBF>line_cbf(2))=line_cbf(2);

MTT= CBV./CBF;
MTT(MTT==inf)=0;
MTT(isnan(MTT)) = 0;
line_mtt=sort(abs(MTT(:)),'DESCEND');
MTT(MTT>line_mtt(15))=line_mtt(15);

[~,TTP]=max(Ct,[],3);
TTP(TTP==inf)=0;
TTP(isnan(TTP)) = 0;
line_ttp=sort(abs(TTP(:)),'DESCEND');
TTP(TTP>line_ttp(20))=line_ttp(20);

end

