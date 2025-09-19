function [r2e] = r2eMap(data, Ktrans, ve, Tc, time)
[w,h,f]=size(data);

r2e=zeros(w,h);

x0=[10];
lb=[0];
ub=[2000];

options = optimset('Display','off');

for wi=1:w
    for hi=1:h
        temp = squeeze(data(wi,hi,:));
        ve_temp = ve(wi,hi);
        Ktrans_temp = Ktrans(wi,hi);
        Tc_temp = Tc(wi,hi);
        if  ve_temp > 0.01 && Ktrans_temp > 0.01
            index_max = find(temp==max(temp));
            % index_max = find(time>Tc_temp,1);
            if index_max > 1
                index = index_max;
                time_temp = time(index:end);
                temp_temp = (temp(index:end));    
                model = @(a, x) a(1)*Ktrans_temp*exp(-(Ktrans_temp./ve_temp).*(x-Tc_temp));
                a = lsqcurvefit(model,x0,time_temp,temp_temp',lb,ub,options);
                r2e(wi,hi)=a(1);
            end
        end
    end
    disp(['Finish line ',num2str(wi)]);
end

end