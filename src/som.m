% read data
trainds = read_dataset_ds('train');
testds = read_dataset_ds('test');

% create feature extractors
alex = alexnet;
alex_size = alex.Layers(1).InputSize;
extract_alex = @(imds) ...
    activations(alex, augmentedImageDatastore(alex_size, imds), ...
                'fc7', 'outputAs', 'columns');
%bag = bagOfFeatures(trainds);
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
[testdata, testclass] = extract_features(testds, extractors);
%load('features.mat');
%load('som.mat');
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
    cw = w/10;
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
        extract = fs{i}
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
