from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import pickle
import operator


import numpy as np


def load_data_set(dat_file_path):
    """
    This method will unpickle the ".dat" file created from music_genre.py which stores the mean_matrix, covariance
    and class (target) for the training set. The training set consists of 10 genres of music, with 100 samples each.
    :param dat_file_path: File path to the ".dat" pickle created by music_genre.py
    :return: A list of length one thousand, a single row will consist of [[mean_matrix], [covariance], class]
    """
    dataset = []
    with open(f"{dat_file_path}", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    return dataset


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


def run(dat_file_path="my.dat", music_directory=os.getcwd() + "\\genres", validation_directory=os.getcwd() + "\\validation\\", num_neighbors=5):
    """
    When run() is called, it will load the given ".dat" file or a default. It will then create a mapping from index to
    genre/folder_name, note: indexing starts at 1. Next it will iterate through the samples in the validation_directory
    create a feature vector for each sample, and predict a class based on the unique counts of its feature-space
    neighbors (num_neighbors)
    :param dat_file_path: Path to the ".dat" pickle file created in music_genre.py
    :param music_directory: Path to the "genre" directory (i.e. the folder containing the classes and training samples).
    :param validation_directory: Path to a directory containing unseen and untested data to validate with.
    :param num_neighbors: Number of neighbors to consider when determining class of sample/input.
    :return: run() will print out each prediction as it takes in samples.
    """
    dataset = load_data_set(f"{dat_file_path}")
    results = {index+1: folder for index, folder in enumerate(next(os.walk(music_directory))[1])}

    for file in os.listdir(validation_directory):
        (rate, sig) = wav.read(validation_directory + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, 0)  # no class for sample input
        pred = nearest_class(get_neighbors(dataset, feature, num_neighbors))
        print(f"The prediction for {file} is: {results[pred]}")


if __name__ == "__main__":
    # select the number of neighbors to consider
    num_neighbors = 5
    # path to required directories
    dat_file_path = os.getcwd() + "\\my.dat"
    validation_directory = os.getcwd() + "\\validation\\"
    music_directory = os.getcwd() + "\\genres"

    run(dat_file_path, music_directory, validation_directory, num_neighbors)
