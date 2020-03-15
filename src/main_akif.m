clc
clear all

%% Load the Folders
imds_train = imageDatastore('train');
imds_test = imageDatastore('test');

%% Getting the labels for training set

len = length(imds_train.Files);
name = string(imds_train.Files(1:len));
label = string(zeros(len,1));
for i = 1:len
    if (strfind(name(i),'airplane')~=0)
        label(i)='Airplane';
    elseif (strfind(name(i),'car')~=0)
        label(i)='Car';
    elseif (strfind(name(i),'cat')~=0)
        label(i)='Cat';
    elseif(strfind(name(i),'dog')~=0)
        label(i)='Dog';
    elseif(strfind(name(i),'flower')~=0)
        label(i)='Flower';
    elseif(strfind(name(i),'fruit')~=0)
        label(i)='Fruit';
    elseif(strfind(name(i),'motorbike')~=0)
        label(i)='Motorbike';  
    else
        label(i)='Person';  
    end
end

imds_train.Labels = categorical(label); % Appending it to the the data store
clearvars i len name label 

%% Getting the labels for test set

len = length(imds_test.Files);
name = string(imds_test.Files(1:len));
label = string(zeros(len,1));
for i = 1:len
    if (strfind(name(i),'airplane')~=0)
        label(i)='Airplane';
    elseif (strfind(name(i),'car')~=0)
        label(i)='Car';
    elseif (strfind(name(i),'cat')~=0)
        label(i)='Cat';
    elseif(strfind(name(i),'dog')~=0)
        label(i)='Dog';
    elseif(strfind(name(i),'flower')~=0)
        label(i)='Flower';
    elseif(strfind(name(i),'fruit')~=0)
        label(i)='Fruit';
    elseif(strfind(name(i),'motorbike')~=0)
        label(i)='Motorbike';  
    else
        label(i)='Person';  
    end
end

imds_test.Labels = categorical(label); % Appending it to the the data store
clearvars i len name label 



%% Data Splitting

[imdsTrain,imdsVal] = splitEachLabel(imds_train,0.8,'randomized'); 

%% Importing the Model and Re-purposing it:
net = alexnet;
inputSize = net.Layers(1).InputSize

layersTransfer = net.Layers(1:end-3); % The last 3 layers were meant for ImageNet hence they are replaced
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Image Augmentation to improve accuracy on Test set and resize images to the model specifications
pixelRange = [-30 30]; % Kept reasonably small as it might be that some of the important features might be on the edges
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ... %Horizontal Reflection 
    'RandXTranslation',pixelRange, ... % Image is laterally displaced along x axis
    'RandYTranslation',pixelRange); % Image is laterally displaced along y axis

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter); % Only Training Images are augmented with Reflection and X-Y Translations 

augimdsVal = augmentedImageDatastore(inputSize,imdsVal); % Resizing to CNN size
augimdsTest = augmentedImageDatastore(inputSize(1:2),imds_test); % Resizing to CNN size

minibatch = preview(augimdsTrain); % Sample of Augmented Images
imshow(imtile(minibatch.input));

%% Training Parameters and Training
options = trainingOptions('adam','InitialLearnRate',1e-4,'ValidationData',augimdsVal,'ValidationFrequency',20,'LearnRateSchedule','piecewise','LearnRateDropPeriod',1,'LearnRateDropFactor',0.75,'MaxEpochs', 5, 'MiniBatchSize', 256, 'Plots','training-progress' ); 
% Experiementally determined parameters. Refer Table 1

netTransfer = trainNetwork(augimdsTrain,layers,options);  

%% Performance Evaluation
[YPred,scores] = classify(netTransfer,augimdsTest) % Accuracy computation
YTest = imds_test.Labels;
accuracy = mean(YPred == YTest)
