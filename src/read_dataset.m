function [samples, classes] = read_dataset(dirpath)
    names = ["airplane", "car", "cat", "dog", ...
             "flower", "fruit", "motorbike", "person"];
    samples = {};
    classes = zeros(0, 1);
    for c = 1:length(names)
        wildcard = sprintf('%s/%s_*.jpg', dirpath, names(c));
        paths = dir(wildcard);
        for p = 1:length(paths)
            pathstr = sprintf("%s/%s", paths(p).folder, paths(p).name);
            samples{end+1} = imread(pathstr);
            classes(end+1, 1) = c;
        end
    end
end
