'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np


def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - a list of k images, the k nearest neighbors of image
    labels - a list of k labels corresponding to the k images
    '''
    neighbors = np.array([])
    labels = []
    kClosestNeighbors = []

    for training_image in train_images:
        distance = np.linalg.norm(image - training_image)
        neighbors = np.append(neighbors, distance)

    kSmallestIndexes = np.argpartition(neighbors, k)

    for value in range(k):
        test = np.array([])
        test = np.append(
            test, train_images[kSmallestIndexes[value]])
        kClosestNeighbors.append(test)

        if train_labels[kSmallestIndexes[value]] == 1:
            labels = np.append(labels, True)
        else:
            labels = np.append(labels, False)

    return np.array(kClosestNeighbors), labels


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []

    for image in dev_images:
        kNearestNeighbor, kLabels = k_nearest_neighbors(
            image, train_images, train_labels, k)
        numTrue = 0
        numFalse = 0
        for element in kLabels:
            if element == True:
                numTrue += 1
            else:
                numFalse += 1

        if numFalse > numTrue or numFalse == numTrue:
            hypotheses.append(0)
            scores.append(numFalse)
        else:
            hypotheses.append(1)
            scores.append(numTrue)

    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    confusions = []

    trueNegative = 0
    falseNegative = 0
    truePositive = 0
    falsePositive = 0

    for index in range(len(hypotheses)):
        if hypotheses[index] == references[index] == 0:
            trueNegative += 1
        elif hypotheses[index] == 0 and references[index] == 1:
            falseNegative += 1
        elif hypotheses[index] == references[index] == 1:
            truePositive += 1
        elif hypotheses[index] == 1 and references[index] == 0:
            falsePositive += 1

    confusions.append([trueNegative, falsePositive])
    confusions.append([falseNegative, truePositive])

    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    accuracy = (truePositive + trueNegative) / (trueNegative +
                                                truePositive + falseNegative + falsePositive)
    f1 = 2 / ((1 / recall) + (1 / precision))

    return np.array(confusions), accuracy, f1
