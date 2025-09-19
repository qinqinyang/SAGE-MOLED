%%
w=128;
h=126;

EXPAND = 256;

total_channel=6;
is_brain=1;

fn='/home/yqq/data_uci/a_SAGE_MOLED/';

infn=[fn,'meas_MID00863_FID1901356_a_sage_oled/'];
target=[fn,'meas_MID00863_FID1901356_a_sage_oled_Charles/'];

if ~exist(target,'dir')==1
    mkdir(target);
end

in_list=dir([infn,'*.mat']);

for jj=1:length(in_list)
    read_fn=[infn,in_list(jj).name];
    load(read_fn);
    
    output = zeros(total_channel,EXPAND,EXPAND);
    img = complex_k2;
    img(:,:,3) = flip(img(:,:,3),2);

    img(:,:,1) = flip(img(:,:,1),2);
    img(:,:,2) = flip(img(:,:,2),2);
    img(:,:,3) = flip(img(:,:,3),2);
    %img = imrotate((img),180);
    
    %norm
    data=img;

    %padding to 256
    expand_2D_complex = zeros(EXPAND, EXPAND, 3) + 1.0i * zeros(EXPAND, EXPAND, 3);
    expand_2D_complex(((EXPAND-w)/2)+1:((EXPAND+w)/2),((EXPAND-h)/2)+1:((EXPAND+h)/2),:)=data;
    
    %ifft
    result_2D_complex=ifft2c(expand_2D_complex);
    
    %norm
    result_2D_complex_norm=result_2D_complex./max(result_2D_complex(:));
    
    %split the real and imag
    j=1;
    output(j,:,:)=real(result_2D_complex_norm(:,:,1));
    j=j+1;
    output(j,:,:)=imag(result_2D_complex_norm(:,:,1));
    j=j+1;
    output(j,:,:)=real(result_2D_complex_norm(:,:,2));
    j=j+1;
    output(j,:,:)=imag(result_2D_complex_norm(:,:,2));
    j=j+1;
    output(j,:,:)=real(result_2D_complex_norm(:,:,3));
    j=j+1;
    output(j,:,:)=imag(result_2D_complex_norm(:,:,3));
    
    % to Charles file

    fn=[target,'Brain256_layer_',num2str(jj,'%03d'),'.Charles'];
    Charles_producer(fn,output,total_channel);
    disp(fn);
end