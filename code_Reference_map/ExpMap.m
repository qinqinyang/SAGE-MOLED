function [t2] = ExpMap(data, TE, mask)
[w,h,e]=size(data);

t2=zeros(w,h);
s0=zeros(w,h);

x0=[100 0.05];
lb=[0 0];
ub=[5000 0.4];

options = optimset('Display','off');
model = @(a, x)a(1)*exp(-(x/a(2)));
parfor wi=1:w
    for hi=1:h
        if  mask(wi,hi) > 0
            temp = squeeze(data(wi,hi,:));
            a = lsqcurvefit(model,x0,TE,temp',lb,ub,options);
            s0(wi,hi)=a(1);
            t2(wi,hi)=a(2);
        end
    end
    disp(['Finish line ',num2str(wi)]);
end

end