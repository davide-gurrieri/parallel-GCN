# Dataset Structure

Each dataset is composed of three files:
- dataset_name.graph
- dataset_name.split
- dataset_name.svmlight

The .graph file contains the edgelist of the current graph

The .split file contains the dataset paritioning with respect to training/validation/testing purposes

The .svmlight contains informations regarding the nodes and the features based on the [SVMLight](https://www.cs.cornell.edu/people/tj/svm_light/) format. 