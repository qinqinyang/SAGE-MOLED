function [aif_cuv,x,y] = AIF_selection_dce(data,TR,showtime,r2)

nSamplesMax = 20; % time point of CA injection

data_r = data;
[nR,nP,nT] = size(data_r);

mask = ones(nR,nP);
[Ct,S0] = caculate_Ct_aif(data_r,mask,nSamplesMax);

%%
figure;imagesc(squeeze(Ct(:,:,showtime)),[0 100]);colormap jet;axis image;
[x,y]=getpts;

points=size(data_r,3);
aif_data=zeros(points, length(x));
for i=1:length(x)
    idx=round(y(i));
    idy=round(x(i));
    aif_data(:,i)=squeeze(Ct(idx,idy,:));
end

aif_cuv=mean(aif_data,2);
r=0.493;
q=2.616;

aif_cuv=(-r+sqrt(r^2+4*q.*aif_cuv))./(2*q);

%%
figure;
plot(aif_cuv);

x = round(x);
y = round(y);
end

function [conc,S0] = caculate_Ct_aif(volumes,mask,nSamplesMax)
% volumes = volumes*1000; % s--->ms
S0=DSC_cal_S0_mean(volumes,mask,nSamplesMax);

conc=zeros(size(volumes));
[nR,nC,nT]=size(volumes);
ind=find(mask);
k=nR*nC;
R0=1./S0;

for t=1:nT
    Rt=1./volumes(:,:,t);
    temp=(Rt-R0).*mask;
    temp(temp<0)=0;
    conc(:,:,t)=temp;
end

% conc=real(conc);
conc(isnan(conc)) = 0;
conc(isinf(conc)) = 0;
end

%% ------------------------------------------------------------------------
function [GVparametri, cv_est_parGV]   = fitGV_picco1(dati,pesi,options_DSC)
% Calcola il fit del primo picco con una funzione gamma-variata.
% La funzione usata � descritta dalla formula:
%
% FP(t)=A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% c(t)=FP(t)
%
% parametri: p=[t0 alpha beta A]
%
% L'ultimo parametro restituito rappresenta l'exitflag, che pu� assumere i
% seguenti valori:
%      1  LSQNONLIN converged to a solution X.
%      2  Change in X smaller than the specified tolerance.
%      3  Change in the residual smaller than the specified tolerance.
%      4  Magnitude search direction smaller than the specified tolerance.
%      5  Voxel nullo
%      0  Maximum number of function evaluations or of iterations reached.
%     -1  Algorithm terminated by the output function.
%     -2  Bounds are inconsistent.
%     -4  Line search cannot sufficiently decrease the residual along the
%         current search direction.

% OPZIONI STIMATORE
options             = optimset('lsqnonlin') ;
options.Display     = 'none'                ;
options.MaxFunEvals = 1000                 ;
options.MaxIter     = 1000                 ;
options.TolX        = 1e-4 ;
options.TolFun      = 1e-4 ;
%options.TolCon      = 1e-2 ;
%options.TolPCG      = 1e-8 ;
options.LargeScale  = 'on' ;
%options.DiffMinChange = 1e-18;
%options.DiffMaxChange = 2  ;

% STIME INIZIALI DEI PARAMETRI (modifica di DENIS)
% Alpha viene impostato a 5
alpha_init=5;

% t0 viene stimato sui dati iniziali. E' calcolato come l'ultimo istante in
% cui i dati rimangono in modulo inferiori al 5% del picco.
[MCdati,TTPpos]=max(dati);
TTPdati=options_DSC.time(TTPpos);
t0_init=options_DSC.time(find(dati(1:TTPpos)<=0.05*MCdati, 1, 'last' ));

% beta viene stimato sfruttando la relazione che TTP=t0+alpha*beta
beta_init=(TTPdati-t0_init)./alpha_init;

% Inizializzo i parametri [t0 alpha beta] e scelgo A in modo che la stima
% iniziale e i dati abbiano lo stesso massimo.
A_init= MCdati./max(GVfunction_picco1([t0_init; alpha_init; beta_init; 1],options_DSC));

% Valori iniziali dei parametri per la stima
% p  = [t0  alpha  beta  A]
p0   = [t0_init;   alpha_init;    beta_init;   A_init] ; % Valori iniziali
lb   = p0.*0.1; % Estremi inferiori
ub   = p0.*10 ; % Estremi superiori

% Controllo i dati, devono essere vettori colonna
if size(options_DSC.time,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    options_DSC.time=options_DSC.time';
end
if size(dati,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    dati=dati';
end
if size(pesi,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    pesi=pesi';
end

% MARCO
% AUMENTO LA PRECISIONE DEL PICCO
[MC TTP]=max(dati);
pesi(TTP)=pesi(TTP)./10;
pesi(TTP-1)=pesi(TTP-1)./2;


%TROVO FINE PRIMO PICCO (20% valore massimo)
i=TTP;
while dati(i)>0.2*dati(TTP)
    i=i+1;
end

%ADATTO I DATI PER "SOLO PRIMO PICCO"
dati_picco1=zeros(size(dati));
dati_picco1(1:i)=dati(1:i);

pesi_picco1=0.01+zeros(size(pesi));
pesi_picco1(1:i)=pesi(1:i);

% STIMATORE
ciclo=true;
nCiclo=0;
p=p0;
while ciclo
    nCiclo=nCiclo+1;
    [p, resNorm, residui, exitFlag,OUTPUT,LAMBDA,JACOBIAN] = lsqnonlin(@objFitGV_picco1, p, lb, ub, options, dati_picco1, pesi_picco1,options_DSC) ;

    if (nCiclo>=4)||(exitFlag>0)
        ciclo=false;
    end
end
GVparametri=p';

J=JACOBIAN;
covp=inv(J'*J);
var=diag(covp);
sd=sqrt(var);

cv_est_parGV=(sd./p*100)';

end

%% ------------------------------------------------------------------------
function [out]                         = objFitGV_picco1(p,dati,pesi,options)
% Funzione obiettivo da minimizzare per la funzione fitGV_picco1
vett=GVfunction_picco1(p,options);

out=(vett-dati)./pesi;
end


%% ------------------------------------------------------------------------
function [GV]                          = GVfunction_picco1(p,options)
% Calcola la funzione gamma-variata definita dai parametri contenuti in p.
% La funzione gamma-variata � definita dalla formula:
%
% GV(t)=A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% parametri: p=[t0 alpha beta A]

t0    = p(1);    % t0
alpha = p(2);    % alpha
beta  = p(3);    % beta
A     = p(4);    % A

nT=length(options.time);
GV=zeros(nT,1);
for cont=1:nT
    t=options.time(cont);
    if t>t0
        GV(cont)=A*((t-t0)^alpha)*exp(-(t-t0)/beta);
    end
end
end


%% ------------------------------------------------------------------------
function [GVparametri, cv_est_parGV]   = fitGV_picco2(dati,pesi,cost_picco1,options_DSC)
% Calcola il fit con una funzione gamma-variata.
% La funzione usata � descritta dalla formula:
%
% FP(t)=A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% c(t)=FP(t) + FP(t-td) conv K*exp(-t/tao)
%
% parametri: p=[t0 alpha beta A td K tao]
%
% L'ultimo parametro restituito rappresenta l'exitflag, che pu� assumere i
% seguenti valori:
%      1  LSQNONLIN converged to a solution X.
%      2  Change in X smaller than the specified tolerance.
%      3  Change in the residual smaller than the specified tolerance.
%      4  Magnitude search direction smaller than the specified tolerance.
%      5  Voxel nullo
%      0  Maximum number of function evaluations or of iterations reached.
%     -1  Algorithm terminated by the output function.
%     -2  Bounds are inconsistent.
%     -4  Line search cannot sufficiently decrease the residual along the
%         current search direction.

% OPZIONI STIMATORE
options             = optimset('lsqnonlin') ;
options.Display     = 'none'                ;
options.MaxFunEvals = 10000                 ;
options.MaxIter     = 10000                 ;
options.TolX        = 1e-8 ;
options.TolFun      = 1e-8 ;
%options.TolCon      = 1e-2 ;
%options.TolPCG      = 1e-8 ;
options.LargeScale  = 'on' ;
%options.DiffMinChange = 1e-18;
%options.DiffMaxChange = 2  ;

% CONTROLLO DEI DATI
% Devono essere tutti vettori colonna
if size(options_DSC.time,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    options_DSC.time=options_DSC.time';
end
if size(dati,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    dati=dati';
end
if size(pesi,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    pesi=pesi';
end
if size(cost_picco1,1)==1
    % controlla che il vettore dei options.time sia un vettore colonna
    cost_picco1=cost_picco1';
end

% PREPARO I DATI PER IL FIT
picco1=GVfunction_picco1(cost_picco1,options_DSC);
dati_picco2=dati-picco1; % I dati da fittare sono i residui del primo fit

%    pesi_picco2=0.01+exp(-dati_picco2); % Pesi per il calcolo del fit.

pesi_picco2=ones(length(dati_picco2),1); % Pesi per il calcolo del fit. (PESI UNIFORMI)
posTaglioPesi=min([find(dati>0.4*max(dati),1,'last'), 3+find(dati==max(dati))]);
% Riduco il peso dei dati prima del picco principale. Arrivo fino a quando
% la concentrazione non � scesa sotto il 40% del picco per evitare casi in
% cui il picco principale non fitta bene i dati successivi al picco e i
% residui potrebbero mostrare un picco fasullo.
pesi_picco2(1:posTaglioPesi)=1;


% INIZIALIZZAZIONE PARAMETRI
% ricerca dei punti iniziali basata sui dati. Considero solo i dati
% dall'istante in cui le concentrazioni scendono sotto il 40% del picco per
% evitare residui troppo rumorosi non relativi al ricircolo. NB: il fit
% viene fatto con tutti i residui.
dati_x_stime_init=dati_picco2;
dati_x_stime_init(1:posTaglioPesi)=0;
dati_x_stime_init(find(dati_x_stime_init<0))=0;

% td_init viene calcolata come distanza tra l'istante del picco principale
% e la distanza del picco del ricircolo. Il picco del ricircolo viene
% individuato come picco dei dati meno la predizione del picco principale.
[maxPicco2,TTPpicco2]=max(dati_x_stime_init);
t0picco2 = find(dati_x_stime_init(1:TTPpicco2)<(0.1*max(dati_x_stime_init)),1,'last');
td_init = options_DSC.time(t0picco2)-cost_picco1(1);

% La stima iniziale di tao � fissata. 100 riesce a dare un ricircolo ben
% spalmato e che con i bound pu� diventare sia una dispersione nulla che
% portare ad una dispersione quasi completa.
tao_init=40;

% La stima di K viene fatta in modo che i massimi del ricircolo predetto e
% dei dati da fittare sia uguale.
ricircolo=GVfunction_ricircolo([cost_picco1; td_init; 1; tao_init],options_DSC);
K_init=max(dati_x_stime_init)./max(ricircolo);


% p  = [td   K   tao]
p = [td_init; K_init; tao_init ] ; % Valori iniziali


% STIMATORE
ciclo=true;
nCiclo=0;
while ciclo
    ub=p.*[10; 10; 10];
    lb=p./[10; 10; 10];
    nCiclo=nCiclo+1;
    [p, resNorm, residui, exitFlag,OUTPUT,LAMBDA,JACOBIAN] = lsqnonlin(@objFitGV_picco2, p, lb, ub, options, dati_picco2,  pesi_picco2,cost_picco1,options_DSC) ;

    if (nCiclo>=4)||(exitFlag>0)
        ciclo=false;
    end
end
GVparametri=p';

J=JACOBIAN;
covp=inv(J'*J);
var=diag(covp);
sd=sqrt(var);

cv_est_parGV=(sd./p*100)';

end

%% ------------------------------------------------------------------------
function [out]                         = objFitGV_picco2(p,dati,pesi,cost_picco1,options)
% Funzione obiettivo da minimizzare per la funzione fitGV
vett=GVfunction_ricircolo([cost_picco1; p],options);

out=(vett-dati)./pesi;
end


%% ------------------------------------------------------------------------
function [ricircolo]                   = GVfunction_ricircolo(p,options)

% Calcola la funzione gamma-variata che descrive il ricircolo delle
% concentrazioni e definita dai parametri contenuti in p.
% La funzione gamma-variata � definita dalla formula:
%
% FP(t)= A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% ricircolo(t)= FP(t-td) conv K*exp(-t/tao)
%
% I parametri sono passati nel sequente ordine:
% p= [t0 alpha beta A td K tao]
%
% NB: dato che la formula prevede una convoluzione, la griglia temporale
% lungo la quale viene calcolata la gamma-variata � molto pi� fitta della
% griglia finale.

t0    = p(1);    % t0
alpha = p(2);    % alpha
beta  = p(3);    % beta
A     = p(4);    % A
td    = p(5);    % td
K     = p(6);    % K
tao   = p(7);    % tao

% 1) Definizione della griglia virtuale necessaria per la convoluzione
TR=options.time(2)-options.time(1);
Tmax=max(options.time);
Tmin=min(options.time);
nT=length(options.time);

TRfine= TR/10;
tGrid=Tmin: TRfine : 2*Tmax;
nTfine=length(tGrid);

% 2) Calcolo le funzioni necessarie per il calcolo del ricircolo
% FP(t)        = A*((t-t0)^alpha)*exp(-(t-t0)/beta)
% disp(t)      = exp(-t/tao)
% ricircolo(t) = K * [FP(t-td) convoluto disp(t)]

% Inizializzazione dei vettori
picco2 = zeros(nTfine,1); % Picco del ricircolo
disp   = zeros(nTfine,1); % Dispersione del ricircolo

for cont=1:nTfine
    t=tGrid(cont);

    if t>t0+td
        % Calcolo di FP(t-td)
        picco2(cont)=K*((t-t0-td)^alpha)*exp(-(t-t0-td)/beta);
    end

    % Calcolo di disp(t)
    disp(cont)=exp(-t/tao);
end

% 3) Assemblo le componenti per ottenere la GV calcolata sulla griglia fine
ricircolo_fine=TRfine.*filter(picco2,1,disp);

% 4) Vado a campionare GV sugli istanti temporali richiesti in options.time
ricircolo=interp1(tGrid,ricircolo_fine,options.time);
end


%% ------------------------------------------------------------------------
function [GV]                          = GVfunction_picco2(p,options)

% Calcola le funzioni gamma-variata che descrivono l'andamento delle
% concentrazioni e definita dai parametri contenuti in p.
% La funzione complessiva � definita dalla formula:
%
% FP(t)= A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% ricircolo(t)= FP(t-td) conv K*exp(-t/tao)
%
% I parametri sono passati nel sequente ordine:
% p= [t0 alpha beta A td K tao]
%
% NB: dato che la formula prevede una convoluzione, la griglia temporale
% lungo la quale viene calcolata la gamma-variata � molto pi� fitta della
% griglia finale.

FP=GVfunction_picco1(p(1:4),options);
ricircolo=GVfunction_ricircolo(p,options);

GV=FP+ricircolo;
end


%% ------------------------------------------------------------------------
function [GV]                          = GVfunction(p,options)
% Calcola la funzione gamma-variata definita dai parametri contenuti in p.
% La funzione gamma-variata � definita dalla formula:
%
% FP(t)=A*((t-t0)^alpha)*exp(-(t-t0)/beta)
%
% c(t)=FP(t) + FP(t-td) conv K*exp(-t/tao)
%
% parametri: p=[t0 alpha beta A td K tao]
% NB: dato che la formula prevede una convoluzione, la griglia temporale
% lungo la quale viene calcolata la gamma-variata � molto pi� fitta della
% griglia finale.

t0    = p(1);    % t0
alpha = p(2);    % alpha
beta  = p(3);    % beta
A     = p(4);    % A
td    = p(5);    % td
K     = p(6);    % K
tao   = p(7);    % tao

% 1) Definizione della griglia virtuale
TR=options.time(2)-options.time(1);
Tmax=max(options.time);
nT=length(options.time);

TRfine= TR/10;
tGrid=0: TRfine : 2*Tmax;
nTfine=length(tGrid);

% 2) Calcolo delle componenti di GV
% Divido la GV nelle sue componenti principali
picco1 = zeros(1,nTfine); % Picco principale
picco2 = zeros(1,nTfine); % Picco del ricircolo
disp   = zeros(1,nTfine); % Dispersione del ricircolo

for cont=1:nTfine
    t=tGrid(cont);

    if t>t0
        picco1(cont)=A*((t-t0)^alpha)*exp(-(t-t0)/beta);
    end

    if t>t0+td
        picco2(cont)=K*((t-t0-td)^alpha)*exp(-(t-t0-td)/beta);
    end

    disp(cont)=exp(-t/tao);
end

% 3) Assemblo le componenti per ottenere la GV calcolata sulla griglia fine
ricircolo=TRfine.*filter(picco2,1,disp);
conc=picco1+ricircolo;

% 4) Vado a campionare GV sui options.time richiesti
GV=zeros(1,nT);
for cont=1:nT
    [err,pos]=min(abs(tGrid-options.time(cont)));
    GV(cont)=conc(pos);

    if err>1
        disp('WARNING: approssimazione non buona.')
    end
end
end
