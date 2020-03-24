addpath('src');

[trainimg, trainclass] = read_dataset('train');

% resize images
w = 100;
n = size(trainimg, 2);
traindata = zeros(w^2, n);
for i = 1:n
    resized = imresize(rgb2gray(trainimg{i}), [w w]);
    traindata(:, i) = resized(:);
end

som_w = 8;
som_h = 8;
x = traindata;
net = selforgmap([som_h som_w]);
net = train(net, x);
y = net(x);
classes = vec2ind(y);

% plot clusters
row = mod(classes-1, som_w)      + 1 + randn(1, n)/5;
col = floor((classes-1) / som_w) + 1 + randn(1, n)/5;
f = figure;
gscatter(row, col, trainclass);
legend('airplane', 'car', 'cat', 'dog', ...
       'flower', 'fruit', 'motorbike', 'person');
saveas(f, 'hej.jpg');

% create classification map
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

% classify training samples
[testimg, testclass] = read_dataset('test');
n = size(testimg, 2);
% resize
testdata = zeros(w^2, n);
for i = 1:n
    resized = imresize(rgb2gray(testimg{i}), [w w]);
    testdata(:, i) = resized(:);
end

x = testdata;
y = net(x);
classes = vec2ind(y);

testclass_guess = zeros(n, 1);
for i = 1:n
    class = map(classes(i), 1);
    testclass_guess(i) = class;
end

err = sum(testclass ~= testclass_guess)
acc = 1 - err/n
