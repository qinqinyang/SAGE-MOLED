function [T2,T2star,s1,s2] = SAGEMap(data, TE)
[w,h,echon]=size(data);

T2star=zeros(w,h);
T2=zeros(w,h);
s1=zeros(w,h);
s2=zeros(w,h);

x0=[100,100,500,500];% t2,t2star,s1,s2
lb=[2,2,10,10];
ub=[1000,1000,4000,4000];

options = optimset('Display','off');

parfor wi=1:w
    for hi=1:h
        temp = squeeze(data(wi,hi,:));
        if  temp(1) > 0
            x = lsqcurvefit(@SAGE_fun,x0,TE,temp',lb,ub,options);
            T2(wi,hi)=1./x(1);
            T2star(wi,hi)=1./x(2);
            s1(wi,hi)=x(3);
            s2(wi,hi)=x(4);
        end
    end
    disp(['Finish line ',num2str(wi)]);
end

end

function y = SAGE_fun(a,t)
    R2=a(1); R2star=a(2); s01=a(3); s02=a(4);
    y=(s01.*exp(-t.*R2star)).*(t<0.045)+(s02.*exp(-0.111.*(R2star-R2)).*exp(-t.*(2.*R2-R2star)).*(t>0.045));
    
end





