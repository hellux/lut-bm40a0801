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

%% Importing the Model and Re-purposing it:
net = alexnet;
inputSize = net.Layers(1).InputSize

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imds_train);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imds_test);

layer= 'fc7'; %fc7, relu7
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');% Extracing Features
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imds_train.Labels;
YTest = imds_test.Labels;
classifier = fitctree(featuresTrain,YTrain); % FIT M-SVM
%classifier = fitcensemble(featuresTrain,YTrain); % FIT M-SVM
YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest)

