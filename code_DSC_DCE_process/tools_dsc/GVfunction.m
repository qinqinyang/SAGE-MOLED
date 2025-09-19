function [GV]= GVfunction(p,options)

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