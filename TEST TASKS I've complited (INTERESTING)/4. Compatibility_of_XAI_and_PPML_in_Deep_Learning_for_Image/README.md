THIS PROJECT WAS EXECUTED USING JUPYTER NOTEBOOK IN COLAB-LIKE APPS (HERE PRODUCED ON KAGGLE)

The whole .ipynd contain outputs to see all results without reproducing them repeatedly

1. Archive contains:
	a) .ipynb file with code

2. To reproduce results one should 
	a) upload these files on google-colab or kaggle
	b) run all cells step by step, reading possible comments above cells or inside them

3. If one wants to train, he should not forget to use GPU to process the notebook. But it takes time


## Description 
The rise of Deep Learning in medical applications raises multiple ethical implications regarding data privacy and trust in the classifier’s explanation. Methods for Privacy-Preserving Machine Learning (PPML) [1] assure that individual patient information remains confidential despite the training of powerful data-driven models, while explainable AI (XAI) aims at elucidating their inner workings [2]. This project aims at benchmarking the impact of Privacy-Preserving Deep Learning on methods of explainable AI (XAI). 

[1] Tanuwidjaja, H. C., Choi, R., & Kim, K. (2019, September). A survey on deep learning techniques for privacy-preserving. In International Conference on Machine Learning for Cyber Security (pp. 29-46). Springer, Cham. 
[2] Ivanovs, M., Kadikis, R., & Ozols, K. (2021). Perturbation-based methods for explaining deep neural networks: A survey. Pattern Recognition Letters. 
## Requirements 
- Git 
- Python 
- PyTorch / TensorFlow / Keras (Preferred TensorFlow) 
- Image Processing 
- Captum (Suggested) 
## Test Task
Download following dataset:
- A subset of the Simple Concept DataBase (SCDB) dataset [here](https://cloud.dfki.de/owncloud/index.php/s/DeHxejXFekTM2KW).

Execute the following tasks for each of the datasets:
1. Train and evaluate an AlexNet model. Compute at least class-wise accuracies.
2. Pick a random sample per class and compute attribution maps with an approach of your choice.
3. Add random noise to the selected input samples. Explore how much random noise is required to significantly change the attribution maps. Evaluate the significance using a correlation coefficient (e.g. Pearson, Spearman or Jaccard) between the original and the modified attribution map.
Include answers to following questions:
1. Name five different attribution methods.
2. Name five different methods applicable to privacy-preserving deep learning.
3. Explain in your own words why privacy-preserving and XAI methods follow competing goals.