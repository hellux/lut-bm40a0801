% read data
trainds = read_dataset_ds('train/');
testds = read_dataset_ds('test/');

% create feature extractors
alex = alexnet;
alex_size = alex.Layers(1).InputSize;
extract_alex = @(imds) ...
    activations(alex, augmentedImageDatastore(alex_size, imds), ...
                'fc7', 'outputAs', 'columns');
%bag = bagOfFeatures(trainds); % time costly
%extract_surf = @(imds) encode(bag, imds)';
extractors = { ...
    extract_alex, ...
%    extract_surf, ...
%    @extract_lbp, ...
%    @extract_grayscale, ...
};

% train classifier
som_w = 8;
som_h = 8;
[traindata, trainclass] = extract_features(trainds, extractors);
[net, map] = som_train(traindata, trainclass, som_w, som_h);

% classify test samples
%load('net/features_alex.mat');
[testdata, testclass] = extract_features(testds, extractors);
[testclass_pred, tiles] = som_classify(testdata, net, map);

% evalutate classification
n = length(testclass);
err = testclass ~= testclass_pred;
acc = 1 - sum(err)/n

% list errors
names = ["airplane", "car", "cat", "dog", ...
         "flower", "fruit", "motorbike", "person"];
for i = 1:n
    if err(i)
        fname = testds.Files{i};
        actual = names(testclass(i));
        pred = names(testclass_pred(i));
        disp(sprintf("%s is %s but classified as %s at (%d, %d)", ...
                     fname, actual, pred, tiles(i, 1), tiles(i, 2)));
    end
end

% plot test dataset clusters
f = figure;
x = tiles(:, 1) + randn(1, n)'/8;
y = tiles(:, 2) + randn(1, n)'/8;
gscatter(x, y, testclass);
legend('airplane', 'car', 'cat', 'dog', ...
       'flower', 'fruit', 'motorbike', 'person');
saveas(f, 'som_clusters.png');

% save feature vectors and classifier to disk
save('features.mat', 'traindata', 'trainclass', 'testdata', 'testclass');
save('som.mat', 'net', 'map');

function features = extract_grayscale(imds)
    n = length(imds.Files);
    w = 100;
    features = zeros(w^2, n);
    for i = 1:n
        I = readimage(imds, i);
        resized_grayscale = imresize(rgb2gray(I), [w w]);
        features(:, i) = resized_grayscale(:);
    end
end

function features = extract_lbp(imds)
    n = length(imds.Files);
    w = 100;
    cw = 16;
    nf = ceil(w/cw)^2 * 59;
    features = zeros(nf, n);
    for i = 1:n
        I = readimage(imds, i);
        resized_grayscale = imresize(rgb2gray(I), [w w]);
        features(:, i) = lbp(resized_grayscale, cw);
    end
end

function [features, class] = extract_features(imds, fs)
    n = length(imds.Files);
    features = zeros(0, n);
    for i = 1:length(fs)
        extract = fs{i};
        features = [features; extract(imds)];
    end

    class = zeros(n, 1);
    names = ["airplane", "car", "cat", "dog", ...
             "flower", "fruit", "motorbike", "person"];
    labels = cellstr(imds.Labels);
    for i = 1:n
        lbl = labels{i};
        idx = find(names==lbl);
        class(i) = idx;
    end
end

function [net, map] = som_train(traindata, trainclass, som_w, som_h)
    n = length(trainclass);

    x = traindata;
    net = selforgmap([som_h som_w]);
    net = train(net, x);
    y = net(x);
    classes = vec2ind(y);

    map = zeros(som_h*som_w, 1);
    for c = 1:som_h*som_w
        counts = zeros(max(trainclass), 1);
        for i = 1:length(classes)
            if classes(i) == c
                counts(trainclass(i)) = counts(trainclass(i)) + 1;
            end
        end
        [~, winner] = max(counts);
        map(c, 1) = winner;
    end
end

function [testclass_pred, tiles] = som_classify(testdata, net, map)
    n = size(testdata, 2);
    x = testdata;
    y = net(x);
    classes = vec2ind(y);

    testclass_pred = zeros(n, 1);
    for i = 1:n
        class = map(classes(i), 1);
        testclass_pred(i) = class;
    end

    w = sqrt(length(map));
    rows = mod(classes-1, w) + 1;
    cols = floor((classes - 1) / w) + 1;
    tiles = [cols' rows'];
end

function hst = lbp(I, cw)
    [h, w] = size(I);
    hst = [];
    for y = 0:ceil(h/cw)-1
        for x = 0:ceil(w/cw)-1
            x_range = 1+cw*x:min(cw*(x+1), w);
            y_range = 1+cw*y:min(cw*(y+1), h);
            hst_cell = lbp_cell(I(y_range, x_range));
            hst = [hst hst_cell];
        end
    end
end

function hst = lbp_cell(I)
    [h, w] = size(I);
    neigh = [0 0; 1 0; 2 0; 2 1; 2 2; 1 2; 0 2; 0 1];
    coeffs = 2.^[0:7];
    uniform = [0 1 2 3 4 6 7 8 12 14 15 16 24 28 30 31 32 48 56 60 62 63 64 ...
        96 112 120 124 126 127 128 129 131 135 143 159 191 192 193 195 199 ...
        207 223 224 225 227 231 239 240 241 243 247 248 249 251 252 253 254 ...
        255];

    hst = zeros(1, 59);
    for y = 1:h
        for x = 1:w
            center = I(y, x);
            number = 0;
            for i = 1:8
                nx = x + neigh(i, 1);
                ny = y + neigh(i, 2);
                if 0 < ny && ny <= h && 0 < nx && nx <= w
                    neigbour = I(ny, nx);
                    if neigbour >= center
                        number = number + coeffs(i);
                    end
                end
            end

            i = find(uniform == number);
            if length(i) == 0
                i = 59;
            end
            hst(1, i) = hst(1, i) + 1;
        end
    end
    hst = hst / sum(hst);
end

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
