fn_dir = 'Template_SMRI_DSC_DCE/';
framen = size(t1_dyn,3);

load('VObj_demo.mat')
VObj.Sus = [];
VObj.B0loc = [];
VObj.B0 = rot90(VObj.B0,3).*mask;
VObj.B1 = rot90(VObj.B1,3).*mask;
for i=1:framen
    VObj.Rho = M0;
    VObj.T1 = t1_dyn(:,:,i);
    VObj.T2 = t2_dyn(:,:,i);
    VObj.T2Star = t2star_dyn(:,:,i);
    save_name = [fn_dir,num2str(i),'.mat'];
    save(save_name,'VObj');
    disp(i);
end