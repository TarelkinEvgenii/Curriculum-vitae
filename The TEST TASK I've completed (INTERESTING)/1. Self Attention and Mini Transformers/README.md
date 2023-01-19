THIS PROJECT WAS EXECUTED USING JUPYTER NOTEBOOK IN COLAB-LIKE APPS (HERE PRODUCED ON KAGGLE)

IF .ipynb DOES NOT CONTAIN CELLS OUTPUTS, 
BUT DON'T WORRY JUST USE STEPS 2.c FOR FAST REPRODUCING THE RESULTS USING GPU WITHOUT REPEATED TRAINING

1. Archive contains 2 files (.ipynb, .pt):
	a) .ipynb file with code
	b) checkpoint of GPT model trained for 5 epochs

2. To reproduce results one should 
	a) upload these files on google-colab or kaggle
	b) run all cells step by step, reading possible comments above cells or inside them
	c) one can upload this .pt file to kaggle root folder (as input, dataset) or google colab by simply dropping it in root and find ceil containing upload method for checkpoint files .pt to. EVEN IN THIS CASE THIS IS RECOMMENDED TO USE GPU IN YOUR NOTEBOOK, BECAUSE THERE ARE CELLS AFTER LOADING THE CHECKPOINT WHICH REQUIRES TIME TO PROCESS

3. If one wants to train, he should not forget to use GPU to process the notebook. But it really takes eternity

## Description of the test task
The aim of the project is to explore the effect of changes to “Self-Attention” Layer in a mini
version of an OpenAI GPT3 Transformer. A mini version called Mini-GPT3 was published
recently and is available at https://github.com/karpathy/minGPT. This mini version can be
trained within 30 minutes on the CIFAR-10 dataset and provides “reasonable” results. This
mini architecture also provides a good opportunity to test out the effect of different
parameters on GPT-3.
## Requirements
● Pytorch (Strong coding skills)
● Python
● Linux
## Test Task
The test task involves training the mini-GPT3 for a few epochs with the below mentioned
changes:
● Read about the OpenAI GPT3 and about self attention
	o https://arxiv.org/abs/2005.14165
	o https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
● Clone the following repo
	o https://github.com/karpathy/minGPT/
	o Open the file play_image.ipynb
● Currently, for training, the images are represented with a Vector Quantization
approach (with a code book size of 512).
● Remove the Vector Quantization approach and instead, use any pre-trained CNN
to convert the image into a vector.
● Now train the mini-GPT for 15-20 epochs (with your new image representation) and
show your generated images.
● Upload your code (+ precise instructions to run your code and recreate your results)
at OLAT in any of the following methods.
	o a Jupyter notebook
	o python files in tar / zip format.
● The whole process should not take you more than ~ 30 mins (excluding the paper
reading part).

NOTE: The aim of the task is not to check if you can generate realistic images and
therefore your submission will NOT be judged based on the quality of generated images.