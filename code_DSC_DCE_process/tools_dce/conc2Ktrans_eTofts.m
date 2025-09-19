function [ktrans,ve,vp]=conc2Ktrans_eTofts(CONC,time,C_AIF,mask)
[np, nv, nt] = size(CONC);

if nargin==2
    C_AIF=SAIF_p(time); % use population-averaged AIF if no AIF input
end

hct = 0.3;
C_AIF = C_AIF(:)./(1-hct);

if mean(diff(time))>0.5
    time=time/60; % convert to minute unit if not
end

[~,peak]=max(C_AIF);  %get rid of the part before peak
peak=peak-1;

%   Compute TK parameters
CONC=CONC(:,:,peak:nt); % only use later points
CONC(find(isnan(CONC)==1)) = 0;
timer = time(peak:nt)';
C_aif=C_AIF(peak:nt);
Cp = C_aif;

%%
[w,h,timei]=size(CONC);
ktrans=zeros(w,h);
ve=zeros(w,h);
vp=zeros(w,h);

parfor wi=1:w
    for hi=1:h
        temp = squeeze(CONC(wi,hi,:));
        if  mask(wi,hi) > 0
            options = fitoptions('Method', 'NonlinearLeastSquares',...
                'MaxIter', 50,...
                'MaxFunEvals', 50,...
                'TolFun', 1e-8,...
                'TolX', 1e-4,...
                'Display', 'off',...
                'Lower',[0.001 0.001 0.001],...
                'Upper', [5 1 1],...
                'StartPoint', [0.005 0.05 0.05],...
                'Robust', 'off');

            ft = fittype('model_extended_tofts_cfit( Ktrans, ve, vp, Cp, T1)',...
                'independent', {'T1', 'Cp'},...
                'coefficients',{'Ktrans', 've', 'vp'});
            [f, gof, output] = fit([timer, Cp],temp,ft, options);
            ktrans(wi,hi) = f.Ktrans;			% ktrans
            ve(wi,hi) = f.ve;				% ve
            vp(wi,hi) = f.vp;				% vp
        end
    end
    disp(['Finish line ',num2str(wi)]);
end

% if output.exitflag<=0
%     % Change start point to try for better fit
%     new_options = fitoptions(options,...
%         'StartPoint', [0.05 0.05 0.05]);
%     [new_f, new_gof, new_output] = fit([timer, Cp'],Ct,ft, new_options);
%
%     if new_gof.sse < gof.sse
%         f = new_f;
%         gof = new_gof;
%         output = new_output;
%         confidence_interval = confint(f,0.95);
%     end
%
%     if output.exitflag<=0
%         % Change start point to try for better fit
%         new_options = fitoptions(options,...
%             'StartPoint', [0.5 0.05 0.05]);
%         [new_f, new_gof, new_output] = fit([timer, Cp'],Ct,ft, new_options);
%
%         if new_gof.sse < gof.sse
%             f = new_f;
%             gof = new_gof;
%             output = new_output;
%             confidence_interval = confint(f,0.95);
%         end
%     end
% end

% toc
% x(1) = f.Ktrans;			% ktrans
% x(2) = f.ve;				% ve
% x(3) = f.vp;				% vp
% x(4) = gof.sse;				% residual
% x(5) = confidence_interval(1,1);% (95 lower CI of ktrans)
% x(6) = confidence_interval(2,1);% (95 upper CI of ktrans)
% x(7) = confidence_interval(1,2);% (95 lower CI of ve)
% x(8) = confidence_interval(2,2);% (95 upper CI of ve)
% x(9) = confidence_interval(1,3);% (95 lower CI of vp)
% x(10) = confidence_interval(2,3);% (95 upper CI of vp)

% residuals = output.residuals;
end

