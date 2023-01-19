import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from dataset import get_strange_symbols_train_loader, get_strange_symbols_test_data, get_strange_symbols_train_data
import torch
import numpy as np
import matplotlib.pyplot as plt


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

class CNNModel_batch_norm(nn.Module):
    def __init__(self):
        super(CNNModel_batch_norm, self).__init__()

        # Convolution 1 (with 2D batch normalisation layer)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2 (with 2D batch normalisation layer)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(32)
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
        out = self.conv1_bn(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # Set 2
        out = self.cnn2(out)
        out = self.conv2_bn(out)
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
    print_confusion_matrix = True
    save_np = True
    learning_rate = 0.01
    n_iters = 10
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

    if(save_np):
        y_predicted_test = model(test_x.float())
        predicted_test = torch.zeros(len(y_predicted_test))
        for j in range(len(predicted_test)):
            predicted_test[j] = torch.argmax(y_predicted_test[j])
        predicted_npy_labels = np.array(predicted_test, dtype=np.int32)
        np.savetxt('test_predicted.csv', predicted_npy_labels, delimiter=',')

    #Printing confusion matrix
    if(print_confusion_matrix):
        train_label_npy=train_label.numpy()
        predicted_npy = np.array(predicted, dtype=np.int32)
        predicted_npy = np.round(predicted_npy)
        print(confusion_matrix(train_label_npy,predicted_npy))

    # Visualize loss and acc
    epoch_for_plot = np.arange(1, n_iters + 1)
    # Visualize loss
    fig, ax = plt.subplots()
    ax.plot(epoch_for_plot, loss_for_plot)
    ax.set(xlabel='Epoch', ylabel='Loss')
    ax.grid()
    plt.savefig("Plot epoch-loss.jpg")
    plt.show()

    # Visualize accuracy
    fig, ax = plt.subplots()
    ax.plot(epoch_for_plot, acc_for_plot)
    ax.set(xlabel='Epoch', ylabel='Accuracy')
    ax.grid()
    plt.savefig("Plot epoch-acc.jpg")
    plt.show()

    # Visualize 10 most confident images per class
    class_labels = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(train_label)):
        if train_label[i]==predicted[i]:
            class_labels[train_label[i]].append(i)
    print("Good")

    most_confident_ten = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    most_confident_ten_confidence = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    ten=10
    confidence = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    confidence_detach = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    for i in range(len(class_labels)):
        # Getting confidence for class_labels
        for k in range(len(class_labels[i])):
            confidence[i].append(torch.softmax(y_predicted[class_labels[i][k]],dim=0)[i])
        for k in range(len(confidence[i])):
            confidence_detach[i].append(confidence[i][k].detach())


        for k in range(10):
            index = np.argmax(confidence_detach[i])
            most_confident_ten[i].append(class_labels[i][index])
            most_confident_ten_confidence[i].append(confidence_detach[i][index])
            confidence_detach[i][index]=0

    for i in range(len(most_confident_ten)):

        fig = plt.figure(figsize=(2, 12), dpi=300, constrained_layout=False, frameon=False)
        fig.suptitle("Class "+str(i+1)+"\nTen correctly classified images\n with maximum confidence ", fontsize=7)
        axs = fig.subplots(nrows=10)

        for j in range(len(most_confident_ten_confidence[i])):
            axs[j].set_title("Confidence " + str(most_confident_ten_confidence[i][j].numpy()), fontsize=4)


        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        for j in range(len(most_confident_ten_confidence[i])):
            axs[j].imshow(train_data[most_confident_ten[i][j],0],cmap='gray')
        plt.savefig("Plot_most_confident_class_"+str(i)+".jpg")
        plt.show()
