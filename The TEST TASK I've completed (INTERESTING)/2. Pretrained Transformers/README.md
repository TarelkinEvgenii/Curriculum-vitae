THIS PROJECT WAS EXECUTED USING JUPYTER NOTEBOOK IN COLAB-LIKE APPS (HERE PRODUCED ON KAGGLE)

The whole .ipynd contain outputs to see all results without reproducing them repeatedly

1. Archive contains:
	a) .ipynb file with code

2. To reproduce results one should 
	a) upload these files on google-colab or kaggle
	b) run all cells step by step, reading possible comments above cells or inside them

3. If one wants to train, he should not forget to use GPU to process the notebook. But it takes time

## Description
Transformer models have been applied to language models as a different alternative for CNNs and RNNs. Those models are trained with large text corpora and reaching high performance. However, Lu et al. found that transformers can be also re-used on different tasks which are not related to text datasets such as bit memory, bit XOR, image classification. In this project, we will focus only on image classification tasks.
Your tasks in this project would be:
	● Evaluate several training strategies, network components, and transformations for
	improving the current state of this approach
	● Datasets: CIFAR-10/100, SVHN, tiny-Imagenet, Imagenette

![[_attachments/Pasted image 20230119073143.png]]

## Reference
[Lu et al., 20201] Lu, Kevin, et al. "Pretrained transformers as universal computation
engines." arXiv preprint arXiv:2103.05247 (2021).
## Requirements
● Python
● pytorch
● pytorch-lightning
● mlflow
● optuna
● matplotlib/seaborn
## Test Task
1. Download a pre-trained ResNet-18 on Imagenet using torchvision
2. Adapt the pre-trained model for CIFAR100 and fine-tuned using SGD (lr=0.1) and
train for 10 epochs with the following configurations
	a. freeze all layers except the first one
	b. freeze all layers except the second one
	c. freeze all layers except the third one
	d. freeze all layers except the fourth one
	e. train using all layers
3. Compare and report your results in a jupyter notebook. You can use the functions
classification_report and confusion_matrix from
https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
4. Upload your answer with a zip file which must contain your code without
the dataset, the jupyter notebook with description, and a README.MD describing
how to run the code and description of each file.