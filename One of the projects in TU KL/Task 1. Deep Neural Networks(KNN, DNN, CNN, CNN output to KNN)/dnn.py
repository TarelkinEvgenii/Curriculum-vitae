# imports a getter for the StrangeSymbol Dataset loader and the test data tensor
import torch
from torch import nn, optim
from dataset import get_strange_symbols_train_loader, get_strange_symbols_test_data
from time import time
import numpy as np

def acc_calculation(predicted, actual):
    return np.mean(predicted == actual)

if __name__ == '__main__':
    # executing this prepares a loader, which you can iterate to access the data
    trainloader = get_strange_symbols_train_loader(batch_size=128)
    train_x, train_y = next(iter(trainloader))
    print('Data is received.')

    # Creating a model
    DNN = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 15),
        nn.ReLU()
    )

    # Loss Function
    loss_func = nn.CrossEntropyLoss()
    #Optimizer
    optimizer = optim.SGD(DNN.parameters(), lr=0.01)

    time_start = time()

    for epoch in range(50):
        training_loss = 0
        optimizer.zero_grad()
        # here we fully iterate the loader each time
        for i, data in enumerate(trainloader):
            i = i  # i is just a counter you may use for logging purposes or such
            img, label = data  # data is a batch of samples, split into an image tensor and label tensor
            imgs = img.view(img.shape[0], -1) #Changing the shape

            prediction = DNN(imgs.float())

            loss = loss_func(prediction, label) #calculate the loss
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        #calculationg the accuracy
        y_predicted = DNN(train_x.view(train_x.shape[0], -1).float())
        predicted = torch.zeros(len(y_predicted))
        for j in range(len(predicted)):
            predicted[j] = torch.argmax(y_predicted[j])

        print('Epoch ', epoch, ' Loss: ', training_loss/len(trainloader), 'Accuracy: ', acc_calculation(predicted.numpy(), train_y.numpy()))
    print('Training is finished. Training time = ',(time()- time_start)/60, 'minutes.')

            # As you may notice, img is of shape n x 1 x height x width, which means a batch of n matrices.
            # But fully connected neural network layers are designed to process vectors. You need to take care of that!
            # Also libraries like matplotlib usually expect images to be of shape height x width x channels.


    # TODO
    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
