clc,clear;

fn = '/home/yqq/data_uci/a_SAGE_MOLED/20240828_动态数据整理/sub30/meas_MID00336_FID1589919_a_sage_oled_1954_IPAT2_Dynamic_SENSE_Charles_T2T2star/';
fn = [fn,'slice_005.mat'];
load(fn);

data = flip(rot90(results(:,:,:,2),2),2); % T2star map
data = abs(rev_tensor(data));
figure;
imshow3(data,[0 0.2]);colormap jet;

nSamples = 20;
showtime = 38;
data_aif = squeeze(data);
[AIF_fit,aifx,aify] = AIF_selection_dsc(data_aif,showtime,nSamples);


