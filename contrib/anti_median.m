function anti_median(pattern, source, dest, s, t, a, b)
    files = dir(sprintf('%s/%s', source, pattern));
    for file = files'
      I = single(imread(sprintf('%s/%s', source, file.name)));
      W = impulsenoiseBlock(I, str2num(s), str2num(t), str2num(a), str2num(b));
      imwrite(W, sprintf('%s/%s', dest, file.name));
    end
end
function result = impulsenoiseBlock(I, s, thr, rangea, rangeb)
%
% function result = impulsenoiseBlock(I,thr)
%
% Author:   Israel Dejene
% Date:     Last updated May 21,2013
% Partner:  Prof. Giulia Boata
% Course:   University of Trento, Data Hiding Course Project
%
% Function   : impulsenoiseBlock
%
% Purpose    : Counter forensic techiques on Median Filtering
%              by adding a a random dithering on graylevels
%
% Parameters : Input gray-scale Image I and threshold parameter thr
%
% Return     : result a gray-scale image which is counter attacked to cancle
%              traces of median filtering
%
% Examples of Usage with threshold level 3
%
%    >> img_MF= medfilt2(IMG,[3 3]);
%    >> img_anti-MF = impulsenoiseBlock(img_MF,3)
%
%
img=double(I);
%% Saturation Blocks
J = stdfilt(I); % ones(5)
fmap=zeros(size(I));
result=I;
[r,c]=size(I);
BZ=2;
radius = floor(s/2);
% one may change the step variable s to 1 to get overlaping blocks (for very
% small image, for instance, 32 * 32)
for x = radius + 1 : s : r - radius
    for y = radius + 1 : s : c - radius
        block = I(x - radius : x + radius, y - radius : y + radius);
        B = block(:)';
        m = median(single(B));
        %% distribution of block median
        a = size(find(B==m),1);
        %% occurace of block center
        b = sum(B == B(floor(s*s/2)+1));
        if(a>=BZ )
            fmap(x - radius : x + radius, y - radius : y + radius)=fmap(x - radius : x + radius, y - radius : y + radius)+1;
        end
        if(b>=BZ )
            fmap(x - radius : x + radius, y - radius : y + radius)=fmap(x - radius : x + radius, y - radius : y + radius)+1;
        end

    end
end
%% random dithering matrix
dit=randi([rangea rangeb],r,c);
%% Adding Dithering ,reduce the feature search in Kirchner,cao
for i=1:r
    for j=1:c
        if(fmap(i,j)==0 || J(i,j)>=thr)
            %a=3;b=11;
            %rn = a + (b-a).*rand(1,1);
            s=round(rand(1));
            if(s==0)
                result(i,j)=result(i,j)+dit(i,j); %rn
            else
                result(i,j)=result(i,j)-dit(i,j); %rn
            end
         end
    end
end
%% More Random Dithering
rr = round(rand(size(img)));
%rara = rr.*round(img)+(1-rr).*(img > rand(size(img)));%not used
%% Result
result=uint8(rr+result);
result=uint8(result);
end
