fn='Data_SE_GRE.mat';
load(fn);

slicen=21;
expand=128;

% TE=[15,30,55,90].*0.001;  % SE
% data = data_se;

TE=[3.6,7.5,12,23,36,50,57.5,65].*0.001; % mGRE
data = data_gre;

T2map=zeros(expand,expand,slicen);
T2starmap=zeros(expand,expand,slicen);

for i=1:21
    MASK =  data(:,:,1,i);
    MASK = MASK./max(MASK(:));
    level = 0.05;
    MASK = im2bw(MASK,level);
    indatafit=squeeze(data(:,:,:,i));
    indatafit = indatafit.*MASK;
    T2map(:,:,i) = ExpMap(indatafit, TE, MASK);
    disp(['Finished slice ',num2str(i)]);
end

idx=[3,5,7,9,11,13,15];
figure;
imshow3(T2map(:,:,idx),[0 0.2],[1,7]);colormap jet;
