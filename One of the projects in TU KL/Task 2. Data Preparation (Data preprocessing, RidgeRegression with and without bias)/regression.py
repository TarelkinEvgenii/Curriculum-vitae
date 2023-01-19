import re
import gensim
import numpy as np
import pandas as pd
import traceback
from gensim.utils import tokenize
from gensim.utils import simple_preprocess
from pandas.io.clipboard import copy
import copy
import sys
import nltk
from sklearn.impute import SimpleImputer
from scipy import stats

import matplotlib.pyplot as plt
from wine_dataset import *
import math
from collections import Counter
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import time
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sklearn


class RidgeRegressionBias:
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        # add the new column with ones in the input matrix
        one_col = np.ones((X.shape[0], 1))  # creation the column with ones
        X = np.c_[one_col, X]  # concatenation column with ones and X matrix
        # call fit of RR
        RidgeRegression.fit(self, X, y)
        return self

    def predict(self, X):
        # add the new column with ones in the input matrix
        one_col = np.ones((X.shape[0], 1))  # creation the column with ones
        X = np.c_[one_col, X]  # concatenation column with ones and X matrix
        # call predict of RR
        RidgeRegression.predict(self, X)
        return self.predict

    def get_omegas(self):
        return RidgeRegression.get_omegas(self)


class RidgeRegression:
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        # It was before
        # dimension = X.shape[1]
        # Correction
        # Putting right dimension
        zero = np.zeros((1))
        if X[0].shape != zero[0].shape:
            dimension = len(X[0])

        else:
            dimension = 1

            # Small test changes
            # X_zeros = np.zeros((len(X), 1), dtype=np.float32)
            # for i, element in enumerate(X):
            #     X_zeros[i] = element
            # X = X_zeros
            # End of small changes

        # End of correction

        CId = np.identity(dimension) * (1 / self.C)
        inv_fact = np.linalg.inv(X.T.dot(X) + CId)
        fact = X.T.dot(y)
        omegas = inv_fact.dot(fact)
        self.omegas = omegas
        return self

    def predict(self, X):
        omegas = self.omegas

        # It was before
        # self.predict = X.dot(omegas)

        # Correction
        if len(omegas) == 1:
            # ndarray(N,) to ndarray(N,1)
            X_zeros = np.zeros((len(X), 1), dtype=np.float32)
            for i, element in enumerate(X):
                X_zeros[i] = element
            self.predict = X_zeros.dot(omegas[0])
        else:
            self.predict = X.dot(omegas)
        # End of correction
        return self.predict

    def get_omegas(self):
        return self.omegas


def plot_str(data, border):
    plt_name = data.name
    # counting the frequence of text values
    counter = Counter(data)
    data_val = counter.keys()
    data_count = counter.values()
    unit_dict = dict(zip(data_val, data_count))  # unity two columns into one dictionary
    sorted_data = sorted(unit_dict.items(), key=lambda item: item[1],
                         reverse=True)  # sorting data according to the values
    list_dict = dict(sorted_data)
    list_dict = {key: val for key, val in list_dict.items() if val > border}  # deleting rows with low frequency
    data_val = list_dict.keys()
    data_count = list_dict.values()
    # Plot histogram
    indexes = np.arange(len(data_val))
    width = 0.7
    plt.xlabel(plt_name)
    plt.ylabel('frequency')
    plt.title('Histogram for ' + plt_name)
    plt.bar(indexes, data_count, width)
    plt.xticks(indexes + width * 0.5, data_val, rotation=90)
    plt.subplots_adjust(bottom=0.45)
    plt.savefig("histogram_" + plt_name + ".jpg")
    plt.show()


def plot_int(data, bin):
    plt_name = data.name
    plt.title('Histogram for ' + plt_name)
    plt.xlabel(plt_name)
    plt.ylabel('frequency')
    print('dictionary: ', data)
    data.hist(bins=bin)
    plt.savefig("histogram_" + plt_name + ".jpg")
    plt.show()


def MSE(true, predict):
    sum = 0
    n = len(true)
    for i in range(0, n):
        difference_of_value = true[i] - predict[i]
        squared_difference = difference_of_value ** 2
        sum = sum + squared_difference
    MSE = sum / n
    return MSE


# return [minimal_value, maximal_value, average_value, standard_deviation, median_value]
def get_statistics(column, indices):
    # Getting_array_without_nan
    numpy_array = column.to_numpy()
    # Column with nan values
    if len(indices) != 0:
        len_of_array_without_nan = len(numpy_array) - len(indices)
        numpy_array_without_nan = np.full(len_of_array_without_nan, np.inf)
        j = 0  # for indices
        k = 0  # for numpy_array_without_nan
        for i, element in enumerate(numpy_array):
            if i != indices[j]:
                numpy_array_without_nan[k] = numpy_array[i]
                k += 1
            else:
                j += 1
                # To overcome out of range error
                if (j == len(indices)):
                    j = len(indices) - 1
    # Column without nan values
    else:
        numpy_array_without_nan = numpy_array
    minimal_value = numpy_array_without_nan[np.argmin(numpy_array_without_nan)]
    maximal_value = numpy_array_without_nan[np.argmax(numpy_array_without_nan)]
    average_value = np.average(numpy_array_without_nan)
    standard_deviation = np.std(numpy_array_without_nan)
    median_value = np.median(numpy_array_without_nan)

    return [minimal_value, maximal_value, average_value, standard_deviation, median_value]


def how_many_missing(column, get_indices=False):
    # Getting only number of nan values
    # math.isnan(x) for finding nan values
    number_of_nan = 0
    if get_indices == False:
        for element in column:
            if (isinstance(element, str) != True):
                if (math.isnan(element)):
                    number_of_nan += 1
        return number_of_nan

    # Getting number of nan values and their indices
    else:
        indices = []
        for i, element in enumerate(column):
            if (isinstance(element, str) != True):
                if (math.isnan(element)):
                    number_of_nan += 1
                    indices.append(i)
        return number_of_nan, indices


def one_hot_encoding(column, is_text, limitation):
    column_len = len(column)
    # initialize as Unsigned integer type (0 to 255)
    new_column = np.zeros((column_len, limitation), dtype=np.float32)
    most = Counter(column).most_common()
    most = np.array(most)
    most = most[:limitation, 0]

    # For columns with numbers
    if is_text == False:
        # most = most.astype(int)
        most = list(most)

        for i in range(column_len):
            print("\r Getting OneHot Vector: " + str(i + 1) + " / " + str(column_len), end="")
            if (isinstance(column.iloc[i], str) != True):
                if (math.isnan(column.iloc[i]) != True):

                    # "Value" in "most"
                    already = 0
                    for j in range(len(most)):
                        if column.iloc[i] == most[j]:
                            new_column[i, j] = 1
                            already = 1
                            break
                    # If "value" not in "most"
                    if already == 0:
                        distance = np.inf
                        indice = np.inf
                        for j in range(len(most)):
                            if distance > abs(column.iloc[i] - most[j]):
                                distance = abs(column.iloc[i] - most[j])
                                indice = j

                        new_column[i, indice] = 1
        print("\r", end="")
        return new_column

    # For columns with text
    elif is_text == True:
        # For columns with text
        most = list(most)

        for i in range(column_len):
            print("\r Getting OneHot Vector: " + str(i + 1) + " / " + str(column_len), end="")
            if (isinstance(column.iloc[i], str) == True):
                for j in range(len(most)):
                    if column.iloc[i] == most[j]:
                        new_column[i, j] = 1
                        break

        print("\r", end="")
        return new_column
    else:
        raise RuntimeError("GROUP_9_CUSTOM_ERROR: Second argument of function should be boolean value")

    print("Good")


def Doc2Vec_our(column, labels, limitation):
    all_words_texts = []
    stops = set(stopwords.words("english"))
    for i in range(len(column)):
        print("\rPreparing " + str(i) + "/" + str(len(column)), end="")
        tokenized_text = []
        sent_text = nltk.sent_tokenize(column.iloc[i])
        for sentence in sent_text:
            # tokenized_text.append(nltk.word_tokenize(sentence))
            tokenized_text.append(simple_preprocess(sentence, deacc=True))
        # to one array
        tokenized_text_all = []
        for i in range(len(tokenized_text)):
            for j in range(len(tokenized_text[i])):
                tokenized_text_all.append(tokenized_text[i][j])
        tokenized_text_all = [word for word in tokenized_text_all if word not in stops]
        all_words_texts.append(tokenized_text_all)
    print("\rAnother preparation...", end="")
    documents = [TaggedDocument(doc, [labels[i]]) for i, doc in enumerate(all_words_texts)]

    print("\rStart learning", end="")
    start_time = time.time()
    model_dv = Doc2Vec(documents, vector_size=limitation, window=2, min_count=1, workers=4)
    print("\r--- %s seconds ---" % (time.time() - start_time))

    print("Start extract vectors", end="")
    vectors = []
    for i in range(len(all_words_texts)):
        b = 0
        for j in range(len(all_words_texts[i])):
            if b == 0:
                sentence = all_words_texts[i][j]
                b = 1
            else:
                sentence = sentence + " " + all_words_texts[i][j]
        print("\rExtracting vectors: " + str(i) + " / " + str(len(all_words_texts)), end="")
        vectors.append(model_dv.infer_vector([sentence]))
    vectors = np.array(vectors)
    print("\r", end="")
    return vectors


def extraction_year(data):
    title = data.loc[:, ('title')]
    year = data.loc[:, ('year')]

    pattern = r'(19\d{2}|200\d{1}|201\d{1}|202\d{1})'
    reg = re.compile(pattern, re.IGNORECASE)

    for row_num in range(len(title)):
        text = title[row_num].split(' ')
        for word in text:
            if word.isdigit():
                if len(word) >= 4:
                    num = re.findall(reg, word)
                    if len(num) != 0:
                        year.iloc[row_num] = int(num[0])
        print("\rGetting years from title: " + str(row_num + 1) + " / " + str(len(title)), end="")
    print()
    # print(year)
    return data


def Reducing_dimension_to_1D(matrix, mode):
    px_coors = []
    lowerer = []
    if mode == 'PCA':
        lowerer.append(PCA(n_components=1))
    if mode == 'TSNE':
        lowerer.append(TSNE(n_components=1))
    if mode == 'combined':
        lowerer.append(PCA(n_components=50))
        lowerer.append(TSNE(n_components=1))

    # Reducing_dimension
    lowed_vecs = matrix.copy()
    for l in lowerer:
        lowed_vecs = l.fit_transform(lowed_vecs)
    px_coors.append(list(list(zip(*lowed_vecs))[0]))
    px_coors = np.array(px_coors[0])
    return px_coors


def cross_validation(clf, X, Y, m=5, metric=MSE, shuffle=True):
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
    if (shuffle == True):
        p = np.random.permutation(n_samples)
        X_Shuffled = X[p]
        Y_Shuffled = Y[p]
    # # # Or without Shuffling
    else:
        X_Shuffled = X
        Y_Shuffled = Y
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

        myclf = clf()
        myclf.fit(X=current_train_set, y=current_train_labels)
        predicted = myclf.predict(X=current_test_set)
        array_of_metrics[i] = metric(predicted, current_test_labels)

    # # # # # Getting average of metrics
    return np.mean(array_of_metrics)


def forward_stepwise(data, labels, restrictions_of_vector_space, num, names):
    empties = 0
    for i in range(len(restrictions_of_vector_space)):
        if restrictions_of_vector_space[i] == 0: empties += 1

    restrictions = np.zeros((len(restrictions_of_vector_space) - empties, 2), dtype=np.int64)
    j = 0
    left = 0
    right = 0
    for i in range(len(restrictions_of_vector_space)):
        if restrictions_of_vector_space[i] != 0:
            if i == 0:
                right = restrictions_of_vector_space[i]
                restrictions[j, 1] = right
            else:
                left = right
                right = right + restrictions_of_vector_space[i]
                restrictions[j, 0] = left
                restrictions[j, 1] = right
            j += 1

    # Deleting
    # del j
    # del left
    # del right
    # del empties

    iteration = 0
    # cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)
    used_items = []
    got_mse = []
    while (iteration < num):
        mse_array = np.full((len(restrictions)), np.inf)
        print("\rIteration " + str(iteration + 1) + " / " + str(num), end="")
        for i in range(len(restrictions)):
            if i not in used_items:
                if iteration == 0:
                    data_for_iter = data[:, restrictions[i, 0]:restrictions[i, 1]]
                    mse_array[i] = cross_validation(RidgeRegressionBias, data_for_iter, labels, shuffle=False)
                else:
                    data_for_iter = np.concatenate(
                        (concatinated_columns, data[:, restrictions[i, 0]:restrictions[i, 1]]), axis=1)
                    mse_array[i] = cross_validation(RidgeRegressionBias, data_for_iter, labels, shuffle=False)

        used_items.append(np.argmin(mse_array))
        got_mse.append(mse_array[np.argmin(mse_array)])

        if iteration == 0:
            concatinated_columns = data[:, restrictions[used_items[0], 0]:restrictions[used_items[0], 1]]
        else:
            # here we should concatinate
            concatinated_columns = np.concatenate((concatinated_columns, data[:, restrictions[used_items[iteration], 0]:
                                                                                 restrictions[
                                                                                     used_items[iteration], 1]]),
                                                  axis=1)

        iteration += 1
    print("\r", end="")

    # got_mse
    # used_items
    for i in range(num):
        string_of_columns = ""
        for j in range(i + 1):
            string_of_columns += (" " + names[used_items[j]])
        print("k=" + str(i + 1) + " MSE=" + str(got_mse[i]) + " formed from columns:" + string_of_columns)


if __name__ == '__main__':
    # START of Initialization
    task_1_c = True  # + plot histograms before preprocessing
    make_a_preprocessing = True  # + with filling all Nan values
    compute_statistics = True  # + inside preprocessing
    Get_vectors = True  # + Task 1E
    task_2_a = True  # + Ridge Regression
    task_2_b_c = True  # + PCA and histograms
    task_2_d = True  # + Cross Validation and MSE
    task_2_e = True  # + Data normalization and Cross-Validation
    task_2_f = True  # + Forward-Stepwise Selection
    task_2_g = True  # + data_2g_labels.csv

    # END of Initialization

    # plot histograms before preprocessing
    if task_1_c == True:
        # OR SAME OPERATION
        data = pd.read_csv('./winemag-data-130k-v2.csv', index_col=0)

        # deleting duplicate rows based on all columns
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)

        # new Add year, but invalid
        data.loc[:, 'year'] = float('nan')
        data_new = extraction_year(data)

        requir = [0, 0, 150, 10, 100, 560, 550, 50, 50, 50, 0, 550, 100, 60]
        columns_int = [3, 4, 13]
        columns_str = [0, 2, 5, 6, 7, 8, 9, 11, 12]

        for i in columns_int:
            col_name = data_new.columns[i]
            print('Column name: ', col_name, '  Type of name: ', type(col_name))
            plot_int(data_new[col_name], requir[i])

        for i in columns_str:
            col_name = data_new.columns[i]
            print('Column name: ', col_name, '  Type of name: ', type(col_name))
            plot_str(data_new[col_name], requir[i])

    if make_a_preprocessing:
        # Read the data
        # data = get_wine_reviews_data()

        # OR SAME OPERATION
        data = pd.read_csv('./winemag-data-130k-v2.csv', index_col=0)

        # deleting duplicate rows based on all columns
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)

        # OLD Add year
        # data.loc[:, 'year'] = float('nan')  # adding the column Year with nan
        # data_new = year_extract(data)

        # new Add year, but invalid
        data.loc[:, 'year'] = float('nan')
        data_new = extraction_year(data)

        # number_of_nan = how_many_missing(data.iloc[:, 4])
        if compute_statistics == True:
            columns_with_numbers = [3, 4, 13]
            j = 0
            for i in range(len(data_new.iloc[0])):
                current_column = data_new.iloc[:, i]
                # for columns with text
                if i != columns_with_numbers[j]:
                    number_of_nan, indices = how_many_missing(current_column, get_indices=True)
                    print("Column " + str(i) + " with name \"" + str(data_new.columns[i]) + "\" has " + str(
                        number_of_nan) + " nan values")
                # for columns with numbers
                else:
                    number_of_nan, indices = how_many_missing(current_column, get_indices=True)
                    print("Column " + str(i) + " with name \"" + str(data_new.columns[i]) + "\" has " + str(
                        number_of_nan) + " nan values")
                    stats = get_statistics(current_column, indices)
                    print("\tMinimal value: " + str(stats[0]) + "\n\tMaximal value: " + str(
                        stats[1]) + "\n\tAverage value: " + str(
                        stats[2]) + "\n\tStandard deviation: " + str(stats[3]) + "\n\tMedian value: " + str(stats[4]))
                    j += 1
                    if j == len(columns_with_numbers):
                        j = len(columns_with_numbers) - 1

        # Since we see that region_2 column has too much NaN values we need to delete this column
        # In order not to worsen the performance of our model

        # Drop 'region_2' column
        data_new = data_new.drop(columns='region_2')
        if ('region_2' in data_new.columns) != True:
            print('\n\'region_2\' column is dropped')

        # Drop column 'title' cause it compiled from several columns column
        data_new = data_new.drop(columns='title')
        if ('title' in data_new.columns) != True:
            print('\'title column\' is dropped')

        # Process the columns ['country'], ['designation'], ['province'], ['region_1'],
        #                           ['taster_name'], ['taster_twitter_handle'], ['variety']
        df_drop_dup_text = data_new[
            ['country', 'designation', 'province', 'region_1', 'taster_name', 'taster_twitter_handle', 'variety']]
        imp = SimpleImputer(strategy="constant")
        df_imp = imp.fit_transform(df_drop_dup_text)
        df_text = pd.DataFrame(df_imp).set_axis(
            ['country', 'designation', 'province', 'region_1', 'taster_name', 'taster_twitter_handle', 'variety'],
            axis=1)
        data_new[['country', 'designation', 'province', 'region_1', 'taster_name', 'taster_twitter_handle',
                  'variety']] = copy.deepcopy(df_text)
        # df_drop_dup = df_drop_dup.drop("Unnamed: 0", axis=1)

        # Process Nans in the column 'price'
        imp = SimpleImputer(strategy='median')
        df_imp1 = imp.fit_transform(data_new['price'].array.reshape(-1, 1))
        df_price = pd.DataFrame(df_imp1).set_axis(['price'], axis=1)
        data_new['price'] = copy.deepcopy(df_price['price'])
        # print("NaNs in a new 'price' column: ", df_price.isna().sum())

        # Process Nans in the column 'year'
        imp = SimpleImputer(strategy='median')
        df_imp2 = imp.fit_transform(data_new['year'].array.reshape(-1, 1))
        df_year = pd.DataFrame(df_imp2).set_axis(['year'], axis=1)
        data_new['year'] = copy.deepcopy(df_year['year'])
        # print("NaNs in a new 'year' column: ", df_year.isna().sum())

        # Prepare some text columns to OneHot encoding
        print("\nPrepare some text columns to OneHot encoding")
        prepare_following_columns_to_one_hot = [0, 5, 6, 7, 8, 9]
        for i in range(len(data_new.iloc[0])):
            if i in prepare_following_columns_to_one_hot:
                for j in range(len(data_new.iloc[:, 0])):
                    print("\rColumn " + str(i) + " Index " + str(j) + "/" + str(len(data_new.iloc[:, 0])), end="")
                    if data_new.iloc[j, i] == "missing_value":
                        data_new.iloc[j, i] = "missing_value " + str(j)
                print("\rColumn " + str(i) + " prepared to OneHot encoding")

        data_new.to_csv("Preprocessed_dataset.csv", sep='\t', encoding='utf-8', index=False)
    else:
        print("Reading CSV")
        data_new = pd.read_csv("Preprocessed_dataset.csv", sep='\t', encoding='utf-8')

    # If you want you can compare 2 arrays
    # data_new_2 = pd.read_csv("Preprocessed_dataset.csv", sep='\t', encoding='utf-8')
    # data_new.compare(data_new_2) # the arrays is equal

    restrictions_of_vector_space = [40, 700, 300, 0, 150, 200, 500, 18, 15, 150, 600, 30]

    # Task 1E - Getting vectors
    if Get_vectors == True:
        print("Start getting vectors")
        # Algorithm to combine columns in one matrix
        type_of_algorithm = [0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 1, 0]  # 0 - OneHot, 1 - Doc2Vec, 3 - None
        Number_or_text = [0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 1]  # 0 - text, 1 - number,3 - None
        first = 0
        for i in range(len(restrictions_of_vector_space)):
            print("Current column for preparation " + str(i) + " / " + str(len(restrictions_of_vector_space)))
            # OneHot
            if type_of_algorithm[i] == 0:
                # text
                if Number_or_text[i] == 0:
                    if first == 0:
                        vectors = one_hot_encoding(data_new.iloc[:, i], True, restrictions_of_vector_space[i])
                        first = 1
                    else:
                        vectors = np.concatenate(
                            (vectors, one_hot_encoding(data_new.iloc[:, i], True, restrictions_of_vector_space[i])),
                            axis=1)
                # number
                else:
                    vectors = np.concatenate(
                        (vectors, one_hot_encoding(data_new.iloc[:, i], False, restrictions_of_vector_space[i])),
                        axis=1)
            elif type_of_algorithm[i] == 1:
                vectors = np.concatenate(
                    (vectors, Doc2Vec_our(data_new.iloc[:, i], data_new.iloc[:, 3], restrictions_of_vector_space[i])),
                    axis=1)

        # Save vectors
        labels = np.array(data_new.iloc[:, 3])
        np.save("Vectors", vectors)
        np.save("Labels", labels)

        print("Finish getting vectors")
    else:
        print("Reading vectors")
        vectors = np.load('Vectors.npy')
        labels = np.load('Labels.npy')

    # Ridge Regression
    if task_2_a == True:
        # splitting into train and test sets
        split_ind = int(np.around(0.8 * vectors.shape[0], decimals=0))
        X, xy = vectors[:split_ind, :], vectors[split_ind:, :]
        y, y_test = labels[:split_ind], labels[split_ind:]

        ridge_model = RidgeRegression()
        ridge_model.fit(X, y)
        prediction_RR = ridge_model.predict(xy)
        print("The Mean Squared Error for Ridge Regression is: " + str(
            MSE(y_test, prediction_RR)) + " (splitting into train and test sets is equal to 80% and 20% respectively)")

    # PCA and histograms
    if task_2_b_c == True:
        # Task 2B
        left = 0
        right = 0
        for i in range(len(restrictions_of_vector_space)):
            if restrictions_of_vector_space[i] != 0:
                if i == 0:
                    right = restrictions_of_vector_space[i]
                else:
                    left = right
                    right = right + restrictions_of_vector_space[i]
                # Reducing dimensionality
                reduced = Reducing_dimension_to_1D(vectors[:, left:right], "PCA")

            plt.scatter(reduced, labels, s=1)
            plt.xlabel("Reduced vector of column")
            plt.ylabel("Points")
            plt.title("Projected value of the column " + "\"" + str(data_new.columns[i]) + "\"")
            plt.savefig('Task_2_Images/Task_2B2_column_' + str(i) + '_' + str(data_new.columns[i]) + '.png')
            plt.show()

            ######################################################################################
            # Computing Ridge Regression
            ridge_model = RidgeRegression()
            ridge_model.fit(reduced, labels)
            prediction = ridge_model.predict(reduced)

            plt.scatter(reduced, labels, s=1, label="Projected value")
            plt.scatter(reduced, prediction, s=1, c='#ff7f0e', label="Value after ridge regression")
            # Regression line
            # the regressed points (w.x+b) on the Y-axis, where x belongs to some range
            omegas = ridge_model.get_omegas()
            omegas = omegas[0]
            min_x = reduced[np.argmin(reduced)]
            max_x = reduced[np.argmax(reduced)]

            plt.plot((min_x, max_x), (min_x * omegas, max_x * omegas), color='green', linewidth=1,
                     label="Regression line")
            # End of Regression line
            # plt.ylim((min_x, max_x))
            plt.xlabel("Reduced vector of column")
            plt.ylabel("Points")
            plt.title("Ridge regression of the column " + "\"" + str(data_new.columns[i]) + "\"")
            plt.legend(loc=0, fontsize=7, framealpha=0.1)
            plt.savefig('Task_2_Images/Task_2B3_column_' + str(i) + '_' + str(data_new.columns[i]) + '.png')
            plt.show()

            ######################################################################################
            # Computing Ridge Regression with bias
            ridge_model_bias = RidgeRegressionBias()
            ridge_model_bias.fit(reduced, labels)
            prediction_bias = ridge_model_bias.predict(reduced)

            plt.scatter(reduced, labels, s=1, label="Projected value")
            plt.scatter(reduced, prediction_bias, s=1, c='#ff7f0e', label="Value after ridge regression with bias")
            # Regression line
            # the regressed points (w.x+b) on the Y-axis, where x belongs to some range
            omegas = ridge_model_bias.get_omegas()
            b = omegas[0]
            w = omegas[1]
            plt.plot((min_x, max_x), (min_x * w + b, max_x * w + b), color='green', linewidth=1,
                     label="Regression line")
            # End of Regression line
            # plt.ylim((min_x, max_x))
            plt.xlabel("Reduced vector of column")
            plt.ylabel("Points")
            plt.title(
                "Ridge regression with bias of the column " + "\"" + str(data_new.columns[i]) + "\"\nVersion_of_plot_1")
            plt.legend(loc=0, fontsize=7, framealpha=0.1)
            plt.savefig('Task_2_Images/Task_2C_column_' + str(i) + '_' + str(data_new.columns[i]) + '(version_1).png')
            plt.show()

            plt.scatter(reduced, labels, s=1, label="Projected value")
            plt.scatter(reduced, prediction_bias, s=1, c='#ff7f0e', label="Value after ridge regression with bias")
            # Regression line
            # the regressed points (w.x+b) on the Y-axis, where x belongs to some range
            omegas = ridge_model_bias.get_omegas()
            b = omegas[0]
            w = omegas[1]
            plt.plot((-10, 10), (-10 * w + b, 10 * w + b), color='green', linewidth=1,
                     label="Regression line")
            # End of Regression line
            # plt.ylim((min_x, max_x))
            plt.xlabel("Reduced vector of column")
            plt.ylabel("Points")
            plt.title(
                "Ridge regression with bias of the column " + "\"" + str(data_new.columns[i]) + "\"\nVersion_of_plot_2")
            plt.legend(loc=0, fontsize=7, framealpha=0.1)
            plt.savefig('Task_2_Images/Task_2C_column_' + str(i) + '_' + str(data_new.columns[i]) + '(version_2).png')
            plt.show()

    # Cross Validation and MSE
    if task_2_d == True:
        # Ridge Regression with bias
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print("In 5-fold cross validation the performance metric MSE is equal = " + str(
            cross_validation(RidgeRegressionBias, vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

    # Data normalization and Cross-Validation
    if task_2_e == True:
        print("Estimate different normalization functions")

        ################### PowerTransformer:yeo-johnson
        print("PowerTransformer:yeo-johnson fit transform")
        start_time = time.time()
        power_transformer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson')
        normilized_vectors = power_transformer.fit_transform(vectors)
        print("\r--- %s seconds ---" % (time.time() - start_time))
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print(
            "In 5-fold cross validation after PowerTransformer:yeo-johnson the performance metric MSE is equal = " + str(
                cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

        ################### QuantileTransformer
        print("QuantileTransformer fit transform")
        start_time = time.time()
        quantile_transformer = sklearn.preprocessing.QuantileTransformer(random_state=0)
        normilized_vectors = quantile_transformer.fit_transform(vectors)
        print("\r--- %s seconds ---" % (time.time() - start_time))
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print("In 5-fold cross validation after QuantileTransformer the performance metric MSE is equal = " + str(
            cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

        ################### MinMaxScaler normalization
        print("MinMaxScaler fit transform")
        start_time = time.time()
        MinMaxScaler_our = sklearn.preprocessing.MinMaxScaler()
        normilized_vectors = MinMaxScaler_our.fit_transform(vectors)
        print("\r--- %s seconds ---" % (time.time() - start_time))
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print(
            "In 5-fold cross validation after MinMaxScaler the performance metric MSE is equal = " + str(
                cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

        ################### Z-score normalization
        normilized_vectors = stats.zscore(vectors)
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print("In 5-fold cross validation after Z-score normalization the performance metric MSE is equal = " + str(
            cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

        ################### L2 normalization
        normilized_vectors = sklearn.preprocessing.normalize(vectors, norm='l2')
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print("In 5-fold cross validation after L2 normalization the performance metric MSE is equal = " + str(
            cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

        ################### L1 normalization
        normilized_vectors = sklearn.preprocessing.normalize(vectors, norm='l1')
        print("Computing 5-fold cross validation with the mean squared error (MSE) as the performance metric.")
        start_time = time.time()
        print("In 5-fold cross validation after L1 normalization the performance metric MSE is equal = " + str(
            cross_validation(RidgeRegressionBias, normilized_vectors, labels, shuffle=False)))
        print("\r--- %s seconds ---" % (time.time() - start_time))

    # Forward-Stepwise Selection
    if task_2_f == True:
        print("Forward-Stepwise Selection: ")
        forward_column = data_new.columns
        forward_column = list(forward_column)
        del forward_column[3]  # deleting points column name
        forward_stepwise(vectors, labels, restrictions_of_vector_space, 5, forward_column)

    # data_2g_labels.csv
    if task_2_g == True:
        # read CSV
        data_2g = pd.read_csv('./data/data_2g.csv', header=None)
        data_2g = np.array(data_2g)
        data_2g = np.reshape(data_2g, [-1])
        data_2g_labels = pd.read_csv('./data/data_2g_labels.csv', header=None)
        data_2g_labels = np.array(data_2g_labels)
        data_2g_labels = np.reshape(data_2g_labels, [-1])

        # Computing Ridge Regression
        ridge_model = RidgeRegression()
        ridge_model.fit(data_2g, data_2g_labels)
        prediction = ridge_model.predict(data_2g)
        print("Ridge Regression without changing data_2g MSE: " + str(MSE(data_2g_labels, prediction)))

        omegas = ridge_model.get_omegas()
        omegas = omegas[0]
        min_x = data_2g[np.argmin(data_2g)]
        max_x = data_2g[np.argmax(data_2g)]

        plt.plot((min_x, max_x), (min_x * omegas, max_x * omegas), color='green', linewidth=1,
                 label="Regression line")

        plt.scatter(data_2g, data_2g_labels, s=1, label="Data points")
        plt.xlabel("Data")
        plt.ylabel("Labels")
        plt.title("Task_2G: Ridge Regression without using fi function")
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Task_2_Images/Task_2G(without_fi).png')
        plt.show()

        # fi=ax^2+bx+c
        # Trying Brute force
        print("Trying Brute force")
        a = np.arange(-10, 10, 0.5)
        b = np.arange(-10, 10, 0.5)
        c = np.arange(-10, 10, 0.5)
        mse_new = np.inf
        mse_current = np.inf
        array_of_parametrs = np.zeros((3))
        done = False
        for i in range(len(a)):
            if done != True:
                for j in range(len(b)):
                    if done != True:
                        for k in range(len(c)):
                            if done != True:
                                fi_data = a[i] * data_2g * data_2g + b[j] * data_2g + c[k]
                                print("\ra=" + str(a[i]) + " b=" + str(b[j]) + " c=" + str(c[k]), end="")
                                mse_new = cross_validation(RidgeRegression, fi_data, data_2g_labels, shuffle=False)
                                if mse_current > mse_new:
                                    array_of_parametrs[0] = a[i]
                                    array_of_parametrs[1] = b[j]
                                    array_of_parametrs[2] = c[k]
                                    mse_current = mse_new
                                    if mse_current < 0.0001:
                                        done = True

        print("\rfi_data=" + "(" + str(array_of_parametrs[0]) + ")" + "*(x^2) + (" + str(
            array_of_parametrs[1]) + ")*x + (" + str(
            array_of_parametrs[2]) + ")")

        print("With our function fi the MSE in Ridge Regression is equal to " + str(
            cross_validation(RidgeRegression, fi_data, data_2g_labels, shuffle=False)))
        print("With our function fi the MSE in Ridge Regression with bias is equal to " + str(
            cross_validation(RidgeRegressionBias, fi_data, data_2g_labels, shuffle=False)))

        ridge_model = RidgeRegression()
        ridge_model.fit(fi_data, data_2g_labels)
        prediction = ridge_model.predict(fi_data)

        omegas = ridge_model.get_omegas()
        omegas = omegas[0]
        min_x = fi_data[np.argmin(fi_data)]
        max_x = fi_data[np.argmax(fi_data)]

        plt.plot((min_x, max_x), (min_x * omegas, max_x * omegas), color='green', linewidth=1,
                 label="Regression line")

        plt.scatter(fi_data, data_2g_labels, s=5, label="Data points")
        plt.xlabel("Data")
        plt.ylabel("Labels")
        plt.title("Task_2G: Ridge Regression after using fi function")
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Task_2_Images/Task_2G(after_fi).png')
        plt.show()

print("Done all tasks")
