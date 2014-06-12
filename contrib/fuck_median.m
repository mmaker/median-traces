function fuck_median(pattern, source, dest, t)
files = dir(sprintf('%s/%s', source, pattern))
   for file = files'
      file.name
      I = single(imread(sprintf('%s/%s', source, file.name)));
      W = impulsenoiseBlock2(I, str2num(t));
      imwrite(W, sprintf('%s/%s', dest, file.name));
    end
end

function result = impulsenoiseBlock2(I, thr)
    img = double(I);
    %% Saturation Blocks
    J = stdfilt(I);

    fmap = zeros(size(I));
    result = I;
    [r, c] = size(I);
    s = 3;
    BZ = 2;
    radius = floor(s/2);
    % one may change the step variable s to 1 to get overlaping blocks (for very
    % small image, for instance, 32 * 32)
    for x = radius + 1 : s : r - radius
        for y = radius + 1 : s : c - radius
            block = I(x - radius : x + radius, y - radius : y + radius);
            B = block(:)';
            m = median(B);
            %% distribution of block median (DBM)
            a = size(find(B == m), 2);
            %% occurace of block center (OBC)
            b = sum(B == B(floor(s*s/2)+1));
            if (a >= BZ )
                fmap(x - radius : x + radius, y - radius : y + radius) = ...
                    fmap(x - radius : x + radius, y - radius : y + radius)+1;
            end
            if (b >= BZ )
                fmap(x - radius : x + radius, y - radius : y + radius) = ...
                    fmap(x - radius : x + radius, y - radius : y + radius)+1;
            end

        end
    end

    %% Adding Dithering ,reduce the feature search in Kirchner,cao
    for i = 1:r
        for j = 1:c
            if (fmap(r,c) == 0 && J(i, j) >= thr)
                a = 3; b = 7;
                rn = a + (b-a) .* rand(1,1);
                result(i,j) = result(i,j)+rn;
            end
        end
    end

    %% More Random Dithering
    rr = round(rand(size(img)));

    %% Result
    result = uint8(rr + result);
    result = uint8(result);

end
