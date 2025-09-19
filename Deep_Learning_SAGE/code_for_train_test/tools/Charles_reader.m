% transform the Charles data to an array
%-----------------------------------------%
% fid_file:file path of Charles data
% w:width of data
% h:height of data
% reture: n*w*h array
%-----------------------------------------%
function H=Charles_reader(fid_file,w,h)
    fid = fopen(fid_file, 'r');
    data_in = fread(fid,'float')';
    step = length(data_in)/w/h;
    H=zeros(step,w,h);
    for n=1:step
        temp_input = data_in(n:step:end);
        H(n,:,:) = reshape(temp_input,[w,h]);
    end
    fclose(fid);
end