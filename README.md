# Running SOM experiments
The SOM experiments can be run by executing the `src/som.m` file. The features
to use can be chosen by changing the `extractors` array by
commenting/uncommenting wanted and unwanted features.

<!--
There are also pre-trained SOM networks in the files `net/som_alex.mat` and
`net/som_lbp.mat` that can be used by commenting out the training part and
loading the `som_xxx.mat` file before the classification. The `extractors`
array also has to match the networks. The `som_alex.mat` only uses the AlexNet
extractor and `som_lbp.mat` only uses the LBP extractor.
-->

# Running CNN experiments
The CNN experiments can be run by executing the
`src/feature_extraction_resnet50.m` and `src/retraining.m`.
<!-- TODO explain further? -->

# Dataset location
By default, the datasets are expected to be located in `train` and `test`
directories. If they are elsewhere the directory has to be specified in the
scripts.
