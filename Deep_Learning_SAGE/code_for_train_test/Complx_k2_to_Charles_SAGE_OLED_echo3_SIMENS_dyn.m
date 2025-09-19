%%
w=128;
h=126;

EXPAND = 256;

total_channel=6;
is_brain=1;

fn='/home/yqq/data_uci/a_SAGE_MOLED/';

infn=[fn,'meas_MID00871_FID1901364_a_sage_oled_dyn/'];
target=[fn,'meas_MID00871_FID1901364_a_sage_oled_dyn_Charles/'];

if ~exist(target,'dir')==1
    mkdir(target);
end

dir_list=dir([infn,'layer*']);

for ii=1:length(dir_list)
    save_dir=[target,'frame_',num2str(ii,'%03d')];
    if ~exist(save_dir,'dir')==1
        mkdir(save_dir);
    end
    dirname=[infn,dir_list(ii).name,'/'];
    file_list=dir([dirname,'*.mat']);

    for jj=1:length(file_list)
        read_fn=[dirname,file_list(jj).name];
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
        fn=[save_dir,'/frame_',num2str(ii,'%03d'),'_slice_',num2str(jj,'%03d'),'.Charles'];
        Charles_producer(fn,output,total_channel);
        disp(fn);
    end
end