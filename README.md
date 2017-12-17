# SemeionNet
 - A set of machine learning experiments with the semeion handwritten digit dataset using Convolutional Neural Network - The objective of this experiment was to determine the optimum number of hidden layers
& related nodes for extracting useful & discriminant variables/features using the semeion handwriting dataset and measure performance of various Convolutional Neural Networks(CNN) architectures. 

## Dataset
 - The semeion dataset is composed of 1593 handwritten digits from 80 persons that were scanned and stretched to a 16x16 size image.
 	- http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit
 - To change the dataset, change the dataset loading code and sample size in the implementation files, if you want to you can also import your own dataset, this code can be easily adapted to classify other type of images.
	```
	width = 16
	height = 16
	dataset = semeion.read_data_semeion()
	```

## Results
 - The results bellow were obtained, using 1115 random entries from the semeio dataset to train the classifier and 478 random entries to test the trained model.
 - Tests were run on a Core i5 4210U CPU with 16GB of RAM.

| CNN Activation Type  | Accuracy |
| -------------------- | -------- |
|      tanh            |  92.05%  |
|     softrelu         |  91.4%   |
|      relu            |  95.30%  |
