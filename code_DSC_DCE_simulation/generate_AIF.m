clc;
%aif
slice=110;
t=1:slice;

num_aif_point=1;
num=num_aif_point;
cr=zeros(num_aif_point,slice);
ca=zeros(num_aif_point,slice);

r=normrnd(2.9,0.05,1,num);
t0=normrnd(2.5,1.5,1,num);
k1=normrnd(0.8,0.05,1,num);
b=normrnd(1.5,0.05,1,num);
td=round(normrnd(8,1,1,num));
tr=randi([15,30],1,num);
k3=randi([400,700],1,num)./10000;

for i=1:num_aif_point
    cm(i,:)=k1(i).*((t-t0(i)).^r(i)).*exp((t0(i)-t)./b(i)).*(t>=t0(i));
    temp_cm=k3(i).*circshift(cm(i,:),td(i));
    temp_cm(1:td(i))=0;
    temp_exp=exp(-t./tr(i));
    temp_cr=conv(temp_cm,temp_exp);
    temp_cr(111:219)=[];
    cr(i,:)=temp_cr;
    ca(i,:)=cm(i,:)+cr(i,:);
end
figure;plot(ca,'LineWidth',2);
Cp=ca;