function Charles_producer(fn,array,num)
[n,w,h]=size(array);
output = zeros(num,w,h);
if n<=num
    for i=1:n
        output(i,:,:)=array(i,:,:);
    end
    [fid,msg]=fopen(fn,'wb');
    fwrite(fid,output,'float');
    fclose(fid); 
end
end