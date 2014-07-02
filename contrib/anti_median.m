function anti_median(pattern, source, dest, s, t, a, b)
    files = dir(sprintf('%s/%s', source, pattern));
    for file = files'
      I = single(imread(sprintf('%s/%s', source, file.name)));
      W = impulsenoiseBlock(I, str2num(s), str2num(t), str2num(a), str2num(b));
      imwrite(W, sprintf('%s/%s', dest, file.name));
    end
end
