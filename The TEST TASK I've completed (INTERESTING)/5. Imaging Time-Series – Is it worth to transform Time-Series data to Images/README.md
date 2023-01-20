This task was done in Kaggle (a Colab-like platform), so you can use one of the similar platforms to run your code.

The .ipynb file provides all the code with all outputs according to the task.

The models do not take a long time to train, so they can be reproduced by simply re-running the training.

## Description
Working with time series data has shown that these data are difficult to interpret. One consequence of this is that a large proportion of publications deal with images and texts, as concepts are more clearly defined there. Consequently, there are significantly more pre-trained models in the field of image analysis. However, it is possible to convert the time series  into images by means of special transformations in order to be able to use the methods existing in the image domain [1]. 
The goal of this project is to find out how this transformation affects performance, robustness and other aspects [2,3] and to what extent it is practical to work with transformed time series instead of the original ones.
[1] Garcia, G. R., Michau, G., Ducoffe, M., Gupta, J. S., & Fink, O. (2021). Temporal signals to images: Monitoring the condition of industrial assets with deep learning image processing algorithms. Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability, 1748006X21994446.
[2] Wang, Z., & Oates, T. (2015, June). Imaging time-series to improve classification and imputation. In Twenty-Fourth International Joint Conference on Artificial Intelligence.
[3] Thanaraj, K. P., Parvathavarthini, B., Tanik, U. J., Rajinikanth, V., Kadry, S., & Kamalanand, K. (2020). Implementation of deep neural networks to classify EEG signals using gramian angular summation field for epilepsy diagnosis. arXiv preprint arXiv:2003.04534.
## Requirements
- Python 3, GIT
- Frameworks: Keras, Tensorflow or PyTorch (preferred PyTorch)
- Suggested Framework: PyTS handles all time-series related stuff
- Theoretical knowledge: CNNs, Model Robustness

## Test Task
For the task you can use jupyter notebook. Try to write the code in a modular way. In every sub-step that is not fully defined write comments that explain your decisions.

The test task id divided in the following sub-tasks:
- Download the CharacterTrajectories dataset e.g. from http://www.timeseriesclassification.com
- Perform the following tasks once for the CharacterTrajectories:
	- Write a Python script to read and appropriately pre-process the data.
	- Write a script to plot one sample of each class.
	- Write a Python script to train AlexNet on the dataset.
	- Evaluate the performance of the classifier for each class.
	- Perform one adversarial attack of your choice and show the prediction difference.