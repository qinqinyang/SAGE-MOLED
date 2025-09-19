fn='Data_SAGE_EPI.mat';
load(fn);

slicen=21;
expand=128;
TE=[19.4,40.2,69.4,90.4,111].*0.001;  % Philips
data = data_sage_philips;

% TE=[20.86,41.96,68.84,89.94,111.04].*0.001; % Siemens
% data = data_sage_siemens;

T2map=zeros(expand,expand,slicen);
T2starmap=zeros(expand,expand,slicen);

for i=1:21
    MASK =  data(:,:,i,1);
    MASK = MASK./max(MASK(:));
    level = 0.05;
    MASK = im2bw(MASK,level);
    indatafit=squeeze(data(:,:,i,:));
    indatafit = indatafit.*MASK;
    [T2,T2star,s1,s2] = SAGEMap(indatafit, TE);
    T2map(:,:,i)=T2;
    T2starmap(:,:,i)=T2star;
    disp(['Finished slice ',num2str(i)]);
end

idx=[3,5,7,9,11,13,15];
figure;
subplot(2,1,1);
imshow3(T2starmap(:,:,idx),[0 0.2],[1,7]);colormap jet;
subplot(2,1,2);
imshow3(T2map(:,:,idx),[0 0.2],[1,7]);colormap jet;