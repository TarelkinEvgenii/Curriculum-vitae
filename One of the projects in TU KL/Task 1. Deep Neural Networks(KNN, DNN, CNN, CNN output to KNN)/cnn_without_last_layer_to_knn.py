import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from dataset import get_strange_symbols_train_loader, get_strange_symbols_test_data, get_strange_symbols_train_data
import torch
import numpy as np
import matplotlib.pyplot as plt
#from knn import KNN,euclidean_distance

def accuracy(predicted, actual):
    return np.mean(predicted == actual)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 15)

    def forward(self, x):
        # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # Dense
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def confusion_matrix(y_true, y_pred):
    # Here you should put only numpy array, not torch tensor,
    # cause this function works without stoping a tensor from tracking history
    # I mean without detach and etc.

    #confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    if len(y_true)!=len(y_pred):
        raise RuntimeError("GROUP_9_CUSTOM_ERROR:Length of y_true and y_pred is different")
    else:
        max_of_true = y_true[np.argmax(y_true)]
        max_of_pred = y_pred[np.argmax(y_pred)]
        confusion_mx = np.zeros((max_of_true+1,max_of_pred+1),dtype=np.int32)
        for i in range(len(y_true)):
            confusion_mx[y_true[i],y_pred[i]]+=1
            print("\rConfusion_matrix: "+str(i+1)+" / "+str(len(y_true)),end="")
        print()
        return confusion_mx

if __name__ == '__main__':

    # executing this prepares a loader, which you can iterate to access the data
    trainloader = get_strange_symbols_train_loader(batch_size=128)

    train_data, train_label = get_strange_symbols_train_data(root='./data/symbols')
    test_x, test_y = get_strange_symbols_test_data(root='./data/symbols')

    model = CNNModel()
    #model = CNNModel_batch_norm()
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.01
    n_iters = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    log_interval = 10
    loss_for_plot = []
    acc_for_plot = []

    print(model)  # print model summary

    # we want to fully iterate the loader multiple times, called epochs, to train successfully
    for epoch in range(n_iters):
        # here we fully iterate the loader each time
        for i, data in enumerate(trainloader):
            i = i  # i is just a counter you may use for logging purposes or such
            img, label = data  # data is a batch of samples, split into an image tensor and label tensor

            # predict = forward pass with our model
            y_predicted = model(img.float())

            # Loss
            l = loss(y_predicted, label)
            # calculate gradients = backward pass
            l.backward()

            # update weights
            optimizer.step()

            # As you may notice, img is of shape n x 1 x height x width, which means a batch of n matrices.
            # But fully connected neural network layers are designed to process vectors. You need to take care of that!
            # Also libraries like matplotlib usually expect images to be of shape height x width x channels.
            # print(img.shape)

        # zero the gradients after updating
        optimizer.zero_grad()
        # if epoch % 10 == 0:
        # [w, b] = model.parameters()  # unpack parameters

        # Getting labels
        y_predicted = model(train_data.float())
        predicted = torch.zeros(len(y_predicted))
        for j in range(len(predicted)):
            predicted[j] = torch.argmax(y_predicted[j])

        print('epoch ', epoch + 1, ': loss = ', l.item(), 'acc = ',
              str(accuracy(predicted.numpy(), train_label.numpy())))

        acc_for_plot.append(accuracy(predicted.numpy(), train_label.numpy()))
        loss_for_plot.append(l.item())



    my_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(my_model)

    data_for_knn = my_model(train_data.float())

    print("Good")
