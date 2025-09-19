function [r2e] = r2eMap_LSQ(data, Ktrans, ve, Tc, time)
[w,h,f]=size(data);

r2e=zeros(w,h);

x0=[10];
lb=[0];
ub=[8000];

options = optimset('Display','off');

for wi=1:w
    for hi=1:h
        temp = squeeze(data(wi,hi,:));
        ve_temp = ve(wi,hi);
        if  temp(1) > 0 && ve_temp > 0
            Tc_temp = Tc(wi,hi);
            index = find(time > 4*Tc_temp, 1);
            time_temp = time(index:end);
            temp_temp = temp(index:end);    
            Ktrans_temp = Ktrans(wi,hi);
            model = @(a, x) a(1).*Ktrans_temp.*exp(-(Ktrans_temp./ve_temp).*(x-Tc_temp));
            a = lsqcurvefit(model,x0,time_temp,temp_temp',lb,ub,options);
            r2e(wi,hi)=a(1);
        end
    end
    disp(['Finish line ',num2str(wi)]);
end

end