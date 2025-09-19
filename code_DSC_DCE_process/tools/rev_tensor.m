function ndata = rev_tensor(data)
[w,h,z] = size(data);
ndata = zeros(w,h,z);
for i=1:z
    ndata(:,:,i) = data(:,:,z-i+1);
end
end