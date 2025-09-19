function [rand_map,rand_small_map]=creat_echo_train_gap(expand_num,max_range,min_range)

rand_map=zeros(expand_num,expand_num);
rand_small_map=zeros(expand_num,expand_num);

xishu_a(1)=0.1; % 0.1---0.4

for i=2:1:11
    xishu_a(i)=rand()-0.5;
end

for i=1:1:11
    xishu_small_a(i)=rand()-0.5;
end

%
if rand()>0.5
    xishu_a(1)= 0.2+0.2*rand();
end


for i=1:1:expand_num
    for j=1:1:expand_num
        
        rand_map(i,j)=0.6+xishu_a(1)+...
            0.5*xishu_a(2)*(i-expand_num/2)/expand_num+...
            0.5*xishu_a(3)*(j-expand_num/2)/expand_num+...
            1*xishu_a(4)*(i-expand_num/2)*(i-expand_num/2)/(expand_num*expand_num)+...
            1*xishu_a(5)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num)+...
            0.5*xishu_a(6)*(i-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num)+...
            0.5*xishu_a(7)*(i-expand_num/2)*(i-expand_num/2)*(i-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_a(8)*(j-expand_num/2)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_a(9)*(i-expand_num/2)*(i-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_a(10)*(i-expand_num/2)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num);
        
        rand_small_map(i,j)=xishu_small_a(1)+...
            0.5*xishu_small_a(2)*(i-expand_num/2)/expand_num+...
            0.5*xishu_small_a(3)*(j-expand_num/2)/expand_num+...
            1*xishu_small_a(4)*(i-expand_num/2)*(i-expand_num/2)/(expand_num*expand_num)+...
            1*xishu_small_a(5)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num)+...
            0.5*xishu_small_a(6)*(i-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num)+...
            0.5*xishu_small_a(7)*(i-expand_num/2)*(i-expand_num/2)*(i-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_small_a(8)*(j-expand_num/2)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_small_a(9)*(i-expand_num/2)*(i-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num)+...
            0.5*xishu_small_a(10)*(i-expand_num/2)*(j-expand_num/2)*(j-expand_num/2)/(expand_num*expand_num*expand_num);
        
    end
end



max_value=max(max(rand_map));
min_value=min(min(rand_map));
rand_map=(rand_map-min_value)/(max_value-min_value);
rand_map=min_range+(max_range-min_range)*rand_map;

max_value=max(max(rand_small_map));
min_value=min(min(rand_small_map));
rand_small_map=(rand_small_map-min_value)/(max_value-min_value);
rand_small_map=min_range+(max_range-min_range)*rand_small_map;

%imshow(rand_small_map,[0,1.4]);colormap jet;
% rand_max_value=0.8*rand()+0.6;
% rand_min_value=0.8*rand()+0.5;
% 
% if rand()>0.5
%     rand_max_value=0.2*rand()+0.7;
%     rand_min_value=0.2*rand()+0.7;
% end
% 
% if rand_max_value < rand_min_value
%     temp_value =rand_max_value;
%     rand_max_value=rand_min_value;
%     rand_min_value=temp_value;
% end
% rand_map=rand_map*(rand_max_value-rand_min_value)+rand_min_value;
% 
% max_value=max(max(rand_small_map));
% min_value=min(min(rand_small_map));
% rand_small_map=0.5+0.9*(rand_small_map-min_value)/(max_value-min_value);
% % 
% % if (rand()>0.3)
% %     rand_small_map=ones(size(rand_small_map));
% % end

end