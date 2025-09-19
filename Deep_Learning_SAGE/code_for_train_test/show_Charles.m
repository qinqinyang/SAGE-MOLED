fn='Deep_Learning_SAGE/data_demo/train/';

filelist=dir([fn,'*.Charles']);

filen=length(filelist);
figure(11);

i =1;
filename=[fn,filelist(i).name];
data=Charles_reader(filename,256,256);
data=permute(data,[2,3,1]);

im1=data(:,:,1);
im2=data(:,:,2);
im=im1+1.0i*im2;

%
subplot(3,3,1);
imshow(abs((im)),[0 0.8]);
subplot(3,3,7);
imshow(abs(fft2c(im)),[0 0.3]);

%
im1=data(:,:,3);
im2=data(:,:,4);
im=im1+1.0i*im2;

subplot(3,3,2);
imshow(abs((im)),[0 0.6]);
subplot(3,3,8);
imshow(abs(fft2c(im)),[0 0.3]);

%
im1=data(:,:,5);
im2=data(:,:,6);
im=im1+1.0i*im2;

subplot(3,3,3);
imshow(abs((im)),[0 0.4]);
subplot(3,3,9);
imshow(abs(fft2c(im)),[0 0.3]);

%
im=data(:,:,7);
figure(11);
subplot(3,3,4);
imshow(abs((im)),[0 0.2]); colormap jet;

im=data(:,:,8);
figure(11);
subplot(3,3,5);
imshow(abs((im)),[0 0.2]); colormap jet;

im=data(:,:,9);
figure(11);
subplot(3,3,6);
imshow(abs((im)),[0 1]); colormap gray;

im=data(:,:,10);
figure(11);
subplot(3,3,7);
imshow((im),[-70 70]);

im=data(:,:,11);
figure(11);
subplot(3,3,8);
imshow(abs((im)),[0.7 1.2]);

im=data(:,:,12);
figure(11);
subplot(3,3,9);
imshow(abs((im)),[0 1]);
