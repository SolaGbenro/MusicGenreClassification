from python_speech_features import mfcc
import scipy.io.wavfile as wav

import os
import pickle
import random
import operator

import numpy as np

CREATE_DAT_FILE = False


def get_neighbors(training_set, instance, k):
    """
    This will return a sorted (closest point at index 0) list of the nearest classes, up to a total of k elements.
    Classes represent music genres.
    :param training_set: List of one thousand tuples storing (feature_vectors, covariance, class). Used to determine
    neighbors of new sample(s).
    :param instance: A new sample to calculate neighbors from. 'instance' is a tuple consisting of
    (np.array(feature_vector), np.array(covariance), class)
    :param k: Number of neighbors to calculate
    :return: A sorted list of classes that has length 'k'
    """
    distances = []
    for x in range(len(training_set)):
        dist = distance(training_set[x], instance, k) + distance(instance, training_set[x], k)
        distances.append((training_set[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    # neighbors is list of the k closest classes to the point/instance.
    return neighbors


def nearest_class(neighbors):
    """
    This method will count each occurrence of each class/genre within the list of neighbors, and return the class/genre
    with the highest count in that list.
    :param neighbors: A sorted list of integers, each digit represents a class/genre of music. The list is sorted by the
    calculated distance (see distance(instance1, instance2, k) method), such that index 0 is the shortest distance
    (closest neighbor), and index -1, is the kth neighbor, the furthest away to make the list.
    :return: The class/genre with the highest occurrence, index 0 is always taken, no ties.
    """
    class_counts = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in class_counts:
            class_counts[response] += 1
        else:
            class_counts[response] = 1

    sorter = sorted(class_counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def get_accuracy(test_set, predictions):
    """
    This method will return the percentage of predictions that were correct when compared with the test set.
    :param test_set: The last element of each row within the test_set list is the class. This is what we compare with
    our list of predictions to determine accuracy.
    :param predictions: List of the predicted classes
    :return: Percentage of correctly guessed predictions within the list
    """
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return 1.0*correct/len(test_set)


def distance(instance1, instance2, k):
    """
    This function is used to determine the distance between two feature vectors.
    :param instance1: First feature vector
    :param instance2: Second feature vector
    :param k: Number of neighbors
    :return: Float value calculation of distance between both vectors
    """
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def create_dat_file(genre_directory=os.getcwd()+"\\genres", dataset="my.dat", verbose=1):
    """
    This method will iterate through each sub-directory or genre of music, and append a feature matrix and class value
    to a .dat file with the given model_name or "my.dat" if none is provided for each sample. This will be used as the
    training set.
    :param genre_directory: Directory holding sub-dirctories with the class or genre as folder names and .wav sample
    files inside
    :param dataset: file name for the newly created .dat pickle file
    :param verbose: With verbose=0, nothing prints. verbose=1, only folder name prints, verbose=2, each file and folder
    name prints
    :return: None. File will be (over)written
    """
    f = open(f"{dataset}", 'wb')

    i = 0
    for folder in next(os.walk(genre_directory))[1]:
        i += 1
        if verbose in [1, 2]:
            print(f"currently working in folder: {folder}")

        for file in os.listdir(genre_directory+folder):
            if verbose == 2:
                print(f"Working on file: {file}")
            (rate, sig) = wav.read(genre_directory+folder+"\\"+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
    f.close()


def load_dataset(dat_file_path, train_test_split=0.70):
    """
    This method will load a given ".dat" pickle file, and return a randomly selected training and testing set, split
    along the provided percentage, or 70% training and 30% testing if nothing is provided.
    :param train_test_split: Percentage of the whole that will be used for training, the difference will be testing.
    :param dat_file_path: Path to the ".dat" pickle file created from training data.
    :return: A randomly selected training and testing set, split along the provided percentage, or 70% training and 30%
    testing if nothing is provided.

    Within these lists, a single row will consist of: [[mean_matrix], [covariance], class]
    """
    dataset = []
    train_set = []
    test_set = []
    with open(f"{dat_file_path}", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for x in range(len(dataset)):
        if random.random() < train_test_split:
            train_set.append(dataset[x])
        else:
            test_set.append(dataset[x])

    return train_set, test_set


if __name__ == "__main__":
    # extract features from the data-set and dump these features into a binary .dat file “my.dat”
    genre_directory = os.getcwd() + r"\\genres\\"
    if CREATE_DAT_FILE:
        create_dat_file(genre_directory, dataset="test.dat", verbose=1)

    # establish training and testing sets
    training_set, test_set = load_dataset("my.dat", 0.66)

    # iterate through test_set creating a list of predictions
    length = len(test_set)
    predictions = []
    for x in range(length):
        predictions.append(nearest_class(get_neighbors(training_set, test_set[x], 5)))

    # calculate accuracy
    accuracy1 = get_accuracy(test_set, predictions)
    print(accuracy1)
