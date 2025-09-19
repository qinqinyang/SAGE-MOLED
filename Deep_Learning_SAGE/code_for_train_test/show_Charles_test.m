fn='Deep_Learning_SAGE/data_demo/test/';

filelist=dir([fn,'*.Charles']);

filen=length(filelist);

i = 2;
idx = 1;
filename=[fn,filelist(i).name];
data=Charles_reader(filename,256,256);
data=permute(data,[2,3,1]);

im1=data(:,:,1);
im2=data(:,:,2);
im=im1+1.0i*im2;
figure(idx);
subplot(2,3,1);
imshow(abs((im)),[0 0.4]);
subplot(2,3,4);
imshow(abs(fft2c(im)),[0 0.5]);

im1=data(:,:,3);
im2=data(:,:,4);
im=im1+1.0i*im2;
figure(idx);
subplot(2,3,2);
imshow(abs((im)),[0 0.4]);
subplot(2,3,5);
imshow(abs(fft2c(im)),[0 0.5]);

im1=data(:,:,5);
im2=data(:,:,6);
im=im1+1.0i*im2;
figure(idx);
subplot(2,3,3);
imshow(abs((im)),[0 0.4]);
subplot(2,3,6);
imshow(abs(fft2c(im)),[0 0.3]);

