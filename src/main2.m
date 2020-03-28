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

%% Bag of Features

bag = bagOfFeatures(imdsTrain);

%% Train a classifier with the Training sets
categoryClassifier = trainImageCategoryClassifier(imdsTrain,bag);
%confMatrix = evaluate(categoryClassifier,imdsVal)%82%

%%
confMatrix = evaluate(categoryClassifier,imds_test)%84
