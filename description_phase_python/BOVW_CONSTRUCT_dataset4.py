"""
The learning consists of:

    1-Extracting local features of all the dataset images
    2-Generating a codebook of visual words with clustering of the features
    3-Aggregating the histograms of the visual words for each of the training images
    4-Feeding the histograms to the classifier to train a model
"""

import cv2
from sklearn.cluster import KMeans
import os
import time
import numpy as np
import pickle
import scipy.cluster.vq as vq

# CONSTANTS
SIFT = cv2.xfeatures2d.SIFT_create()
TRAINING_DATASET_FOLDER = 'images/vehicles_split/minitraining'
DETECTOR = 'SIFT'
DESCRIPTOR = 'SIFT'
K_PARAMETER = 800  # IN BOVW we are talking about the visual words


def prepareFiles(rootpath):
    files_names = []
    ids = []
    labels = []
    current_gt_id = 0
    for root, subfolder, files in os.walk(rootpath, topdown=False):
        for name in files:
            files_names.append(root + '/' + name)
            ids.append(current_gt_id)
            labels.append(name.split('/')[-1])
        current_gt_id += 1
    return files_names, ids, labels


def features(file_name, extractor):
    image = cv2.imread(file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = extractor.detectAndCompute(gray, None)
    return keypoints, descriptors


def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def getDatasetFeatures(file_names):
    print('Start getDatasetFeatures...')
    init = time.time()
    dataset_descriptors = []
    for n_filename in file_names:
        keypoints, descriptors = features(n_filename, SIFT)
        dataset_descriptors.append(descriptors)
        print('extracting features from ' + n_filename)
    # dataset descriptors concatenation
    np_dataset_descriptors = np.array(dataset_descriptors[0])
    training_dataset_images = dataset_descriptors.__len__()
    for n_image in range(1, training_dataset_images):
        np_element = np.array(dataset_descriptors[n_image])
        np_dataset_descriptors = np.concatenate((np_dataset_descriptors, np_element), axis=0)
        print('adding ' + str(np_element.shape[0]) + ' descriptors to create code block')
    end = time.time()
    print('Features extraction ended in ' + str("{0:.3f}".format(end - init)) + ' secs.')
    return keypoints, descriptors, np_dataset_descriptors


def getAndSaveCodeBlock(np_dataset_descriptors, k_parameter,codebook_filename):
    print('Start getCodeBlock with'+ str(np_dataset_descriptors.shape[0]) + 'samples...')
    # code block creation
    init = time.time()
    code_book = KMeans(n_clusters=k_parameter)
    code_book.fit(np_dataset_descriptors)
    end = time.time()
    print('kmeans fitted in ' + str(end - init) + ' secs.')

    pickle.dump(code_book, open(codebook_filename, "wb"))
    return code_book


def getAndSaveBoVWRepresentation(np_dataset_descriptors, k_parameter, codebook,visual_words_filename):
    print('Extracting visual word representations')
    init = time.time()
    visual_words = np.zeros((len(np_dataset_descriptors), k_parameter), dtype=np.float32)
    for i in range(len(np_dataset_descriptors)):
        words, distance = vq.vq(np_dataset_descriptors[i], codebook)
        visual_words[i, :] = np.bincount(words, minlength=k_parameter)
    end = time.time()
    print('Done in ' + str(end - init) + ' secs.')
    pickle.dump(visual_words, open(visual_words_filename, "wb"))
    return visual_words


# preparing list of file names,labels an identifiers for training dataset images  training
Training_dataset_Filenames, Ids_trainingSet, Labels_trainingSet = prepareFiles(TRAINING_DATASET_FOLDER)

# features extraction
Training_KPoints, Training_Descriptors, Training_Np_Descriptors = getDatasetFeatures(Training_dataset_Filenames)
Codebook_filename = 'CB_' + DETECTOR + '_' + DESCRIPTOR + '_' + str(Training_Np_Descriptors.shape[0]) \
                        + 'samples_'+ str(K_PARAMETER) + 'centroids.dat'
Visual_words_filename = 'VW_train_' + DETECTOR + '_' + DESCRIPTOR + '_' + str(Training_Np_Descriptors.shape[0]) \
                        +  'samples_' + str(K_PARAMETER) + 'centroids.dat'


# code block creation
CodeBook = getAndSaveCodeBlock(Training_Np_Descriptors, K_PARAMETER,Codebook_filename)

# Load previously generated codebook comment for new codebook generation
#  CodeBook = pickle.load(open(Codebook_filename, 'r'))
#  print('codebook loaded from'+ str(Codebook_filename))

# get visual words
getAndSaveBoVWRepresentation(Training_Descriptors , K_PARAMETER, CodeBook,Visual_words_filename)