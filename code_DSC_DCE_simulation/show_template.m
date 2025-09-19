figure;
subplot(3,3,1);
imshow(t1,[0 3.5]);title('T1');
subplot(3,3,2);
imshow(t2,[0 0.2]);title('T2');
subplot(3,3,3);
imshow(t2star,[0 0.2]);title('T2star');
subplot(3,3,4);
imshow(M0,[0 1]);title('M0');
subplot(3,3,5);
imshow(CBV,[0 15]);title('CBV');
subplot(3,3,6);
imshow(CBF,[0 100]);title('CBF');
subplot(3,3,7);
imshow(ktrans,[0 0.2]);title('Ktrans');
subplot(3,3,8);
imshow(ve,[0 0.15]);title('Ve');
subplot(3,3,9);
imshow(vp,[0 0.04]);title('Vp');
colormap jet;

%%
figure;
subplot(1,3,1);
imshow(ktrans,[0 0.4]);title('Kt');colorbar;
subplot(1,3,2);
imshow(vp,[0 0.04]);title('Vp');colorbar;
subplot(1,3,3);
imshow(ve,[0 0.3]);title('Ve');colorbar;
colormap jet;

%%
figure;
subplot(1,2,1);
imshow(CBV,[0 15]);title('CBV');
subplot(1,2,2);
imshow(CBF,[0 100]);title('CBF');
colormap jet;