clear,clc;

load("data_for_conv.mat");
AIF = AIF';

results_1 = filter(R_temp,1,AIF);

figure;
subplot(1,3,1);plot(AIF);title("AIF");
subplot(1,3,2);plot(R_temp);title("R(t)");
subplot(1,3,3);plot(results_1);title("fileter(AIF,R)");


results_2 = conv_inter(R_temp',AIF');

figure;
subplot(1,3,1);plot(AIF);title("AIF");
subplot(1,3,2);plot(R_temp);title("R(t)");
subplot(1,3,3);plot(results_2);title("fileter(AIF,R)");

function Y=conv_inter(x,h)
m=length(x);
n=length(h);
for i=1:n
    Y(i)=0;
    for j=1:m
        if(i-j+1>0)
            Y(i)=Y(i)+x(j)*h(i-j+1);
        else
        end
    end
end
end