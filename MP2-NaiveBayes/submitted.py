'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter
from copy import deepcopy
import math

stopwords = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren", "'t", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he", "he", "'d", "he", "'ll", "he", "'s", "her", "here", "here", "hers", "herself", "him", "himself", "his", "how", "how", "i", "'m", "'ve", "if", "in", "into", "is", "isn", "it", "its", "itself", "let", "'s", "me", "more", "most", "mustn", "my", "myself", "no", "nor",
                "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan", "she", "she", "'d", "she", "ll", "she", "should", "shouldn", "so", "some", "such", "than", "that", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "there", "these", "they", "they", "they", "they", "'re", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn", "we", "we", "we", "we", "we", "'ve", "were", "weren", "what", "what", "when", "when", "where", "where", "which", "while", "who", "who", "whom", "why", "why", "with", "won", "would", "wouldn", "you", "your", "yours", "yourself", "yourselves"])


def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''

    mergedWordsNeg = []
    mergedWordsPos = []

    for entry in train['neg']:
        for word in entry:
            mergedWordsNeg.append(word)

    c = Counter(mergedWordsNeg)

    for entry in train['pos']:
        for word in entry:
            mergedWordsPos.append(word)

    d = Counter(mergedWordsPos)

    return {'neg': c, 'pos': d}


def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''

    copy = deepcopy(frequency)
    for item in stopwords:
        if (item in copy['pos'].keys()):
            del copy['pos'][item]

    return copy


def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.

    '''

    result = {
        'neg': {

        },
        'pos': {

        }
    }

    numAllWordsInNeg = 0
    typeWordsInNeg = 0

    numAllWordsInPos = 0
    typeWordsInPos = 0

    for word in nonstop['neg'].keys():
        numAllWordsInNeg += nonstop['neg'][word]
        if (nonstop['neg'][word] >= 1):
            typeWordsInNeg += 1

    for word in nonstop['pos'].keys():
        numAllWordsInPos += nonstop['pos'][word]
        if (nonstop['pos'][word] >= 1):
            typeWordsInPos += 1

    testNeg = smoothness / (numAllWordsInNeg +
                            smoothness * (typeWordsInNeg + 1))
    testPos = smoothness / (numAllWordsInPos +
                            smoothness * (typeWordsInPos + 1))
    result['neg']['OOV'] = testNeg
    result['pos']['OOV'] = testPos

    for word in nonstop['neg'].keys():
        result['neg'][word] = (nonstop['neg'][word] + smoothness) / \
            (numAllWordsInNeg + smoothness * (typeWordsInNeg + 1))

    for word in nonstop['pos'].keys():
        result['pos'][word] = (nonstop['pos'][word] + smoothness) / \
            (numAllWordsInPos + smoothness * (typeWordsInPos + 1))

    return result


def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    result = []

    for text in texts:
        classIsPositive = math.log(prior)
        classIsNegative = math.log(1 - prior)

        for word in text:
            if (word not in stopwords):
                if (word not in likelihood['pos']):
                    classIsPositive += math.log(likelihood['pos']['OOV'])
                else:
                    classIsPositive += math.log(likelihood['pos'][word])

                if (word not in likelihood['neg']):
                    classIsNegative += math.log(likelihood['neg']['OOV'])
                else:
                    classIsNegative += math.log(likelihood['neg'][word])

        if (classIsNegative > classIsPositive):
            result.append("neg")
        elif (classIsNegative < classIsPositive):
            result.append("pos")

    return result


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text

    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword

    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''

    result = np.zeros((len(priors), len(smoothnesses)))

    for priorValue in range(len(priors)):
        for smoothValue in range(len(smoothnesses)):
            priorsEntry = priors[priorValue]
            smoothEntry = smoothnesses[smoothValue]

            smoothingResult = laplace_smoothing(nonstop, smoothEntry)
            hypotheses = naive_bayes(texts, smoothingResult, priorsEntry)

            count_correct = 0

            for x in range(len(labels)):
                if (labels[x] == hypotheses[x]):
                    count_correct += 1

            result[priorValue, smoothValue] = count_correct / len(labels)

    return result
