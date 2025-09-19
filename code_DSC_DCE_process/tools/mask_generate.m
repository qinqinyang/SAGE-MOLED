function mask = mask_generate(data1,thr)
data1=data1./max(max(data1));
B1=[ 0 1 0;1 1 1;0 1 0];
B2=strel('disk',1);

T = thr; % needs to adjust

%T = graythresh(data1);
BW1 = imbinarize(data1,T);
BW2=bwareaopen(BW1,1000);
BW3=imdilate(BW2,B2);
BW3=imfill(BW3,'holes');
BW4=imerode(BW3,B2);
BW5=imclose(BW4,B1);
mask = BW5;
end