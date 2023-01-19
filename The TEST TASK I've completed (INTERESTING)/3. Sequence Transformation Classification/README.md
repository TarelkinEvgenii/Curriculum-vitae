This task was done in Kaggle (a Colab-like platform), so you can use one of the similar platforms to run your code.

The .ipynb file provides all the code with all outputs according to the task.

The models do not take a long time to train, so they can be reproduced by simply re-running the training.

## Description
Image classification is not considered a standard sequence classification task. Biswas et
al. proposed a series of transformations (e.g., applied on the same image in order to
generate a sequence. With this in mind, the newly created sequence is used as an input
sample for an LSTM model. The goal of this project is to extend this idea to other
scenarios.
Your tasks in this project would be:
	● Evaluate several training strategies, network components, and transformations for
	improving the current state of this approach
	● Exploiting CNN networks for improving this approach via distillation knowledge
	● Datasets: CIFAR-10/100, SVHN, tiny-Imagenet, Imagenette
## References
[Biswas2014] Biswas, S. et al. “Using recurrent networks for non-temporal classification
tasks”. IJCNN2014
[Hochreiter1997] Hochreiter, S., and Schmidhuber, J. "Long short-term memory." Neural
computation 1997.
[Pascanu2013] Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty
of training recurrent neural networks." International conference on machine learning. 2013.
## Requirements
● Python
● pytorch
● pytorch-lightning
● mlflow
● optuna
● matplotlib/seaborn

## Test Task
1. Train the following model on CIFAR10
	a. Bidirectional LSTM (hidden_size: 32) + 2 fully connected layers (hidden size: 100)
2. Generate an identity sequence (e.g. repeat the input several times)
	a. Reshape the input image from 3x32x32 (3 dims) to 3072 (1 dims)
	b. Read the Lambda Transformation from torchvision (https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Lambda). One example how to use the Lambda Transformation is found in the Five Crop Transformation (https://pytorch.org/vision/stable/transforms.html#torchvision.transforms FiveCrop)
	c. Evaluate several lengths of the sequence
	i. length = 5
	ii. length = 10
	iii. length = 20
3. Compare and report your results in a jupyter notebook. You can use the functions
classification_report and confusion_matrix from https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
4. Upload your answer with a zip file which must contain your code without
the dataset, the jupyter notebook with description, and a README.MD describing
how to run the code and description of each file.