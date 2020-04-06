function imds = read_dataset_ds(dirpath)
    imds = imageDatastore(dirpath);
    n = length(imds.Files);
    labels = string(zeros(n, 1));
    for i = 1:n
        fname = imds.Files{i};
        spos = strfind(fname, '/');
        dpos = strfind(fname, '_');
        labels(i) = fname(spos(end)+1:dpos(end)-1);
    end
    imds.Labels = categorical(labels);
end
