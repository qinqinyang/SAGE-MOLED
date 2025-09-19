%% FXLStep1AAIF, vp
function Ct = model_extended_tofts_dsc_in(r2,vp, Cp, T1)

Cp = Cp(:);
T1 = T1(:);

% Pre-alocate for speed
Ct = zeros(1,numel(T1));
for k = 1:numel(T1)
    Ct(k) = r2.*vp*Cp(k);
end

Ct = Ct';