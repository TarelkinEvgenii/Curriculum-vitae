# KNN file
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from dataset import *
from datetime import datetime


def euclidean_distance(a, b):
    image_1 = a.reshape(-1)
    image_2 = b.reshape(-1)
    distance = np.square(image_1 - image_2)  # (x_j-x_j')**2 for every pixel in the image
    distance = np.sum(distance)  # adds all distances
    distance = np.sqrt(distance)
    return distance


def manhattan_distance(a, b):
    image_1 = a.reshape(-1)
    image_2 = b.reshape(-1)
    distance = np.abs(image_1 - image_2)
    distance = np.sum(distance)
    return distance


def chebyshev_distance(a, b):
    image_1 = a.reshape(-1)
    image_2 = b.reshape(-1)
    distance = np.abs(image_1 - image_2)
    distance = distance[np.argmax(distance)]
    return distance


class KNN:
    def __init__(self, k=5, dist_function=euclidean_distance):
        self.k = k
        self.dist_function = dist_function
        self.train_data = []
        self.train_labels = []

    def fit(self, X, y):
        """
        Train the k-NN classifier.

        :param X: Training inputs. Array of shape (n, ...)
        :param y: Training labels. Array of shape (n,)
        """
        if (len(X) == len(y)):
            # if (len(self.train_data) == 0 and len(self.train_labels) == 0):
            self.train_data = np.copy(X)
            self.train_labels = np.copy(y)
            # else:
            #     print("\nRetraining the classifier")
            #     added_elements = 0
            #     for j in range(len(X)):
            #         add = 0
            #         # Comparing element-wise
            #         for i in range(len(self.train_data)):
            #             if ((self.train_data[i] == X[j]).all()):
            #                 add = 1
            #                 break
            #         # If element is new, then add it to our BD
            #         if (add == 0):
            #             X_add = np.expand_dims(X[j], axis=0)
            #             self.train_data = np.concatenate((self.train_data, X_add))
            #             y_add = np.expand_dims(y[j], axis=0)
            #             self.train_labels = np.concatenate((self.train_labels, y_add))
            #             added_elements = added_elements + 1
            #         if (j % 150 == 0 and j != 0):
            #             print("\rTraining process " + str(round((j / len(X)) * 100)) + "%", end="")
            #     print("\nEnd of training\nAdded " + str(added_elements) + " knowledge-elements")

        else:
            raise RuntimeError("GROUP_9_CUSTOM_ERROR:Length of input data and input labels has different size")

    def predict(self, X, get_indices=False):
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """

        predictions = np.ones((len(X)))
        # Computing predictions
        for i in range(len(predictions)):
            distances_array = np.ones(len((self.train_data)))
            # Computation distances
            for j in range(len(distances_array)):
                distances_array[j] = self.dist_function(X[i], self.train_data[j])
            # Computing indices of k smallest elements
            indices = np.ones((self.k), dtype=np.int32)
            for j in range(len(indices)):
                indices[j] = np.argmin(distances_array)
                distances_array[indices[j]] = np.inf
            if (get_indices == True):
                return indices
            # Computing most common
            labels_array = np.ones_like(indices)
            for j in range(len(labels_array)):
                labels_array[j] = self.train_labels[indices[j]]
            most_common = Counter(labels_array).most_common()

            predictions[i] = most_common[0][0]

            if (i % 10 == 0):
                print("\rGetting predictions " + str(i) + " of " + str(len(predictions)), end="")
        return predictions


class weighted_KNN:
    def __init__(self, k=5, dist_function=euclidean_distance):
        self.k = k
        self.dist_function = dist_function
        self.train_data = []
        self.train_labels = []

    def fit(self, X, y):
        """
        Train the k-NN classifier.

        :param X: Training inputs. Array of shape (n, ...)
        :param y: Training labels. Array of shape (n,)
        """
        if (len(X) == len(y)):
            # if (len(self.train_data) == 0 and len(self.train_labels) == 0):
            self.train_data = np.copy(X)
            self.train_labels = np.copy(y)
            # else:
            #     print("\nRetraining the classifier")
            #     added_elements = 0
            #     for j in range(len(X)):
            #         add = 0
            #         # Comparing element-wise
            #         for i in range(len(self.train_data)):
            #             if ((self.train_data[i] == X[j]).all()):
            #                 add = 1
            #                 break
            #         # If element is new, then add it to our BD
            #         if (add == 0):
            #             X_add = np.expand_dims(X[j], axis=0)
            #             self.train_data = np.concatenate((self.train_data, X_add))
            #             y_add = np.expand_dims(y[j], axis=0)
            #             self.train_labels = np.concatenate((self.train_labels, y_add))
            #             added_elements = added_elements + 1
            #         if (j % 150 == 0 and j != 0):
            #             print("\rTraining process " + str(round((j / len(X)) * 100)) + "%", end="")
            #     print("\nEnd of training\nAdded " + str(added_elements) + " knowledge-elements")

        else:
            raise RuntimeError("GROUP_9_CUSTOM_ERROR:Length of input data and input labels has different size")

    def predict(self, X):
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """

        predictions = np.ones((len(X)))
        # Computing predictions
        for i in range(len(predictions)):
            distances_array = np.ones(len((self.train_data)), dtype=np.float32)
            # Computation distances
            for j in range(len(distances_array)):
                distances_array[j] = 1 / self.dist_function(X[i], self.train_data[j])
            # Computing indices of k smallest elements
            indices = np.ones((self.k), dtype=np.int32)
            for j in range(len(indices)):
                indices[j] = np.argmin(distances_array)
                distances_array[indices[j]] = np.inf
            # Computing most common
            labels_array = np.ones_like(indices)
            for j in range(len(labels_array)):
                labels_array[j] = self.train_labels[indices[j]]
            most_common = Counter(labels_array).most_common()

            predictions[i] = most_common[0][0]

            if (i % 10 == 0):
                print("\rGetting predictions " + str(i) + " of " + str(len(predictions)), end="")
        return predictions


def accuracy(predicted, actual):
    return np.mean(predicted == actual)


def cross_validation(clf, X, Y, m=5, metric=accuracy):
    """
    Performs m-fold cross validation.

    :param clf: The classifier which should be tested.
    :param X: The input data. Array of shape (n, ...).
    :param Y: Labels for X. Array of shape (n,).
    :param m: The number of folds.
    :param metric: Metric that should be evaluated on the test fold.
    :return: The average metric over all m folds.
    """

    n_samples = len(Y)
    # # # Catching errors
    if m > n_samples:
        raise ValueError(
            (
                "Cannot have number of splits n_splits={0} greater"
                " than the number of samples: n_samples={1}."
            ).format(m, n_samples)
        )

    # # # Shuffle data
    p = np.random.permutation(n_samples)
    X_Shuffled = X[p]
    Y_Shuffled = Y[p]

    # # # Data splitting
    # param m: The number of folds.
    X_folds = []
    Y_folds = []
    number_in_fold = int(n_samples / m)
    for i in range(m):
        if (i < m - 1):
            X_folds.append(X_Shuffled[i * number_in_fold:(i + 1) * number_in_fold])
            Y_folds.append(Y_Shuffled[i * number_in_fold:(i + 1) * number_in_fold])
        elif (i == m - 1):
            X_folds.append(X_Shuffled[i * number_in_fold:])
            Y_folds.append(Y_Shuffled[i * number_in_fold:])

    # # # Getting metrics
    array_of_metrics = np.zeros(m)
    for i in range(m):
        current_train_set = []
        current_train_labels = []
        current_test_set = []
        current_test_labels = []
        g = 0
        # Filling sets
        for j in range(m):
            if i != j:
                if g == 0:
                    current_train_set = np.asarray(X_folds[j])
                    current_train_labels = np.asarray(Y_folds[j])
                    g = g + 1
                else:
                    current_train_set = np.concatenate((current_train_set, np.asarray(X_folds[j])))
                    current_train_labels = np.concatenate((current_train_labels, np.asarray(Y_folds[j])))
                    g = g + 1
            else:
                current_test_set = np.asarray(X_folds[j])
                current_test_labels = np.asarray(Y_folds[j])

        clf.fit(X=current_train_set, y=current_train_labels)
        predicted = clf.predict(X=current_test_set)
        array_of_metrics[i] = metric(predicted, current_test_labels)

    # # # # # Getting average of metrics
    return np.mean(array_of_metrics)


def plot_gt(best_k, searched_index, name, prediction, train_x, train_y, i_indices_of_false):
    plot_name = name
    plot_ncols = best_k + 1
    fig = plt.figure(figsize=(3, 1), dpi=300, constrained_layout=False, frameon=False)
    axs = fig.subplots(ncols=plot_ncols)
    fig.suptitle(plot_name, fontsize=5)
    # axs.set(title='KNN Euclidean')
    axs[0].set_title("Predicted label: " + str(prediction[searched_index]) + "\nGT: " + str(
        train_y[searched_index]), fontsize=5)
    for i in range(plot_ncols - 1):
        axs[i + 1].set_title("\nGT: " + str(train_y[i_indices_of_false[0][i]]), fontsize=5)
        # Deleting visible border of the frame
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    axs[0].imshow(train_x[best_k, 0], cmap='gray')
    for i in range(plot_ncols - 1):
        axs[i + 1].imshow(train_x[i_indices_of_false[0][i], 0], cmap='gray')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(plot_name + ".jpg")
    plt.show()


########################################from OXANA's code

# Convalutional calculation
def convolve2D(image, kernel, padding=1, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    x_kern_shape = kernel.shape[0]
    y_kern_shape = kernel.shape[1]
    x_img_shape = image.shape[0]
    y_img_shape = image.shape[1]

    # Shape of Output Convolution
    x_output = int(((x_img_shape - x_kern_shape + 2 * padding) / strides) + 1)
    y_output = int(((y_img_shape - y_kern_shape + 2 * padding) / strides) + 1)
    output = np.zeros((x_output, y_output))

    # Apply Equal Padding to All Sides
    if padding != 0:
        image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(image_padded)
    else:
        image_padded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - y_kern_shape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - x_kern_shape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * image_padded[x: x + x_kern_shape, y: y + y_kern_shape]).sum()
                except:
                    break

    return output


########################################from OXANA's code


def main(args):
    # Set up data
    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()
    train_y = np.array(train_y)

    test_x, test_y = get_strange_symbols_test_data(root=args.train_data)
    test_x = test_x.numpy()
    # test_y is empty

    # DONE: Load and evaluate the classifier for different k

    # # # # # # Performs m-fold cross validation for KNN
    knn_k = np.arange(1, 11, 1)
    knn_accuracy = []
    for i, k in enumerate(knn_k):
        # some why it works only with lists
        print("\nUsing different k: " + str(k) + " / " + str(len(knn_k)))
        knn_accuracy.append(cross_validation(KNN(k=k), train_x, train_y))
    np.save('knn_accuracy', np.array([knn_accuracy]))
    #

    knn_accuracy = np.array(knn_accuracy)

    best_k = knn_k[np.argmax(knn_accuracy)]

    # 0 KNN Eucledian best
    # 1 KNN best_k Manhattan_distance
    # 2 KNN best_k Chebyshev distance
    # 3 Weighted KNN best
    # 4 Gaussian Blur Filter
    # 5 Edge Detection Filter
    # 6 Concatenation Filters

    different_accuracies = np.zeros((7))
    different_accuracies[0] = knn_accuracy[np.argmax(knn_accuracy)]
    different_accuracies[1] = cross_validation(KNN(k=best_k, dist_function=manhattan_distance), train_x,
                                               train_y)
    different_accuracies[2] = cross_validation(KNN(k=best_k, dist_function=chebyshev_distance), train_x,
                                               train_y)

    # # # # # # Performs m-fold cross validation for KNN with Weights
    print("\n\nKNN with Weights")
    weighted_knn_accuracy = []
    for i, k in enumerate(knn_k):
        # some why it works only with lists
        print("\nUsing different k: " + str(k) + " / " + str(len(knn_k)))
        weighted_knn_accuracy.append(cross_validation(weighted_KNN(k=k), train_x, train_y))
    np.save('weighted_knn_accuracy', np.array([weighted_knn_accuracy]))

    different_accuracies[3] = weighted_knn_accuracy[np.argmax(weighted_knn_accuracy)]

    ######################################## from OXANA's code
    # Filters
    # Gaussian Blur Filter
    kernel0 = np.array(([1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]), dtype="float") * (1.0 / 16)
    # Sharpen Filter
    kernel1 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    # Edge Detection Filter
    kernel2 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

    # Bottom Sobel Filter
    kernel3 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # Concatinate 4 different feature maps
    train_labels_max = train_y[np.argmax(train_y)]
    kernels = [kernel0, kernel2]
    ######################################## from OXANA's code

    ######################################## my addition to OXANA's code
    # # # # # Getting feature representations
    list_of_all_features = []
    for i in range(len(train_y)):
        list_of_features = [[], []]
        for j in range(len(kernels)):
            image = train_x[i, 0]
            list_of_features[j] = convolve2D(image, kernels[j], padding=1, strides=1)
        list_of_all_features.append([list_of_features])
        print("\rGetting feature maps for images " + str(i + 1) + " / " + str(len(train_y)), end="")
    print()
    list_of_all_features = np.array(list_of_all_features)  # 15000,1,2,28,28
    # list_of_all_features[:,:,0]    15000,1,28,28 # Gaussian Blur Filter
    # list_of_all_features[:,:,1]    15000,1,28,28 # Edge Detection Filter
    # train_x                        15000,1,28,28
    # (list_of_all_features[:,:,0][0,0]==list_of_all_features[0,0,0]).all() TRUE (0 image)
    # (list_of_all_features[:,:,0][:,0]==list_of_all_features[:,0,0]).all() TRUE (all images)

    # Application the KNN algorithm on the features separately as well as the concatenation of all features.

    concatinated_list_of_features = np.concatenate((list_of_all_features[:, :, 0], list_of_all_features[:, :, 1]),
                                                   axis=2)

    different_accuracies[4] = cross_validation(KNN(k=best_k), list_of_all_features[:, :, 0],
                                               train_y)  # 4 Gaussian Blur Filter
    different_accuracies[5] = cross_validation(KNN(k=best_k), list_of_all_features[:, :, 1],
                                               train_y)  # 5 Edge Detection Filter
    different_accuracies[6] = cross_validation(KNN(k=best_k), concatinated_list_of_features,
                                               train_y)  # 6 Concatenation Filters

    ######################################## my addition to OXANA's code

    # DONE: Plot results

    # # # # # # Visualize the accuracy of different k
    fig, ax = plt.subplots()
    ax.plot(knn_k, knn_accuracy, label='KNN Euclidean distance with different k\nBest Euclidean acc: ' + str(
        round(float(different_accuracies[0]) * 100) / 100))
    ax.plot(knn_k, weighted_knn_accuracy,
            label='KNN with weights (Euclidean) with different k\nBest Euclidean acc: ' + str(
                round(float(different_accuracies[3]) * 100) / 100))
    ax.plot(best_k, different_accuracies[1], 'ro',
            label='KNN Manhattan distance acc: ' + str(round(float(different_accuracies[1]) * 100) / 100))
    ax.plot(best_k, different_accuracies[2], 'bo',
            label='KNN Chebyshev distance acc: ' + str(round(float(different_accuracies[2]) * 100) / 100))
    ax.plot(best_k, different_accuracies[4], 'go',
            label='KNN Gaussian Blur Filter (Euclidean) acc: ' + str(round(float(different_accuracies[4]) * 100) / 100))
    ax.plot(best_k, different_accuracies[5], 'mo',
            label='KNN Edge Detection Filter (Euclidean) acc: ' + str(
                round(float(different_accuracies[5]) * 100) / 100))
    ax.plot(best_k, different_accuracies[6], 'ko',
            label='KNN Concatenation Filters (Euclidean) acc: ' + str(
                round(float(different_accuracies[6]) * 100) / 100))
    ax.legend()
    ax.set(xlabel='Different K', ylabel='Accuracy',
           title='Visualize the accuracy of different k')
    ax.set_ylim(0, 1.5)
    ax.grid()
    plt.savefig(("Accuracy_of_different_k" + str(datetime.now().strftime('%M_%S')) + ".jpg"))
    plt.show()

    # # # # # Plot different missclassified images
    # Task I
    # KNN Ground truth best_k Euclidean distance
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Euclidean'
    model = KNN(k=best_k)
    model.fit(X=train_x, y=train_y)
    i_predictions = model.predict(X=train_x)


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(model.predict(X=np.expand_dims(train_x[searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, train_x, train_y, i_indices_of_false)

    # KNN Ground truth best_k Manhattan_distance
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Manhattan'
    model = KNN(k=best_k, dist_function=manhattan_distance)
    model.fit(X=train_x, y=train_y)
    i_predictions = model.predict(X=train_x)


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(model.predict(X=np.expand_dims(train_x[searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, train_x, train_y, i_indices_of_false)

    # KNN Ground truth besk_k chebyshev_distance
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Chebyshev'
    model = KNN(k=best_k, dist_function=chebyshev_distance)
    model.fit(X=train_x, y=train_y)
    i_predictions = model.predict(X=train_x)


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(model.predict(X=np.expand_dims(train_x[searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, train_x, train_y, i_indices_of_false)

    # KNN Ground truth besk_k Euclidean distance (Gaussian Blur Filter)
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Euclidean - Blur Filter'
    model = KNN(k=best_k)
    model.fit(X=list_of_all_features[:, :, 0], y=train_y)
    i_predictions = model.predict(X=list_of_all_features[:, :, 0])


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(
            model.predict(X=np.expand_dims(list_of_all_features[:, :, 0][searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, list_of_all_features[:, :, 0], train_y,
            i_indices_of_false)

    # KNN Ground truth besk_k Euclidean distance (Edge Detection Filter)
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Euclidean - Edge Detection Filter'
    model = KNN(k=best_k)
    model.fit(X=list_of_all_features[:, :, 1], y=train_y)
    i_predictions = model.predict(X=list_of_all_features[:, :, 1])


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(
            model.predict(X=np.expand_dims(list_of_all_features[:, :, 1][searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, list_of_all_features[:, :, 1], train_y,
            i_indices_of_false)
    # KNN Ground truth best_k Euclidean distance (Concatenation Filters)
    i_indices_of_false = []
    searched_index = np.NAN
    plot_name = 'KNN Concatenation Filters'
    model = KNN(k=best_k)
    model.fit(X=concatinated_list_of_features, y=train_y)
    i_predictions = model.predict(X=concatinated_list_of_features)


    my_prediction = np.array(i_predictions)
    for i in range(len(my_prediction)):
        if my_prediction[i] != train_y[i]:
            searched_index = i
            break

    if (searched_index != np.NAN):
        i_indices_of_false.append(
            model.predict(X=np.expand_dims(concatinated_list_of_features[searched_index], axis=0), get_indices=True))
    plot_gt(best_k, searched_index, plot_name, my_prediction, concatinated_list_of_features, train_y,
            i_indices_of_false)

    # # # # # Visualize a plot containing a few images from each class # # # # #
    train_labels_max = train_y[np.argmax(train_y)] + 1

    fig = plt.figure(figsize=(3, 15))
    subfigs = fig.subfigures(15, 1)

    for j in range(train_labels_max):

        bb = 0
        axs = subfigs[j].subplots(ncols=3)
        axs[1].set_title("Class " + str(j + 1), fontsize=14)
        # Deleting visible border of the frame
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        for i in range(len(train_y)):
            if (train_y[i] == j):
                axs[bb].imshow(train_x[i, 0], cmap='gray')
                bb = bb + 1
            if (bb == 3):
                break

    plt.savefig("Plot_contains _images_of_all_classes.jpg")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='This script computes cross validation scores for a kNN classifier.')

    parser.add_argument('--folds', '-m', type=int, default=5,
                        help='The number of folds that the data is partitioned in for cross validation.')
    parser.add_argument('--train-data', type=str, default=DEFAULT_ROOT,
                        help='Directory in which the training data and the corresponding labels are located.')
    parser.add_argument('--k', '-k', type=int, default=list(range(1, 11)) + [15], nargs='+',
                        help='The k values that should be evaluated.')

    args = parser.parse_args()
    main(args)
