'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    maxNumWord0 = 0
    maxNumWord1 = 0

    for text in texts:
        word0Counter = 0
        word1Counter = 0
        for word in text:
            if word == word0:
                word0Counter += 1
            elif word == word1:
                word1Counter += 1
        if maxNumWord0 < word0Counter:
            maxNumWord0 = word0Counter
        elif maxNumWord1 < word1Counter:
            maxNumWord1 = word1Counter

    pjoint = np.zeros((maxNumWord0 + 1, maxNumWord1 + 1))

    for text in texts:
        word0Counter = 0
        word1Counter = 0
        for word in text:
            if word == word0:
                word0Counter += 1
            elif word == word1:
                word1Counter += 1
        pjoint[word0Counter, word1Counter] += 1

    pjoint2 = pjoint / 500

    return pjoint2


def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    temp = []

    if index == 0:
        for text in Pjoint:
            tempEntry = 0
            for prob in text:
                tempEntry += prob
            temp.append(tempEntry)

    else:
        Pjoint = Pjoint.T
        for text in Pjoint:
            tempEntry = 0
            for prob in text:
                tempEntry += prob
            temp.append(tempEntry)
    return temp


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pjoint = Pjoint / Pmarginal[:, None]

    return Pjoint


def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    '''
    temp = np.average(P)

    return temp


def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    '''
    avg = 0
    counter1 = 0
    for entry in P:
        avg += counter1 * entry
        counter1 += 1

    result = 0
    counter = 0

    for entry in P:
        tempResult = ((counter - avg) ** 2) * (entry)
        result += tempResult
        counter += 1

    return result


def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    '''

    expectXY = 0

    for row in range(len(P)):
        for column in range(len(P[row])):
            print(P[row][column])
            expectXY += row * column * P[row][column]

    expectValueX = marginal_distribution_of_word_counts(P, 0)
    expectValueY = marginal_distribution_of_word_counts(P, 1)

    xAvg = 0
    yAvg = 0

    for value in range(len(expectValueX)):
        xAvg += value * expectValueX[value]

    for value in range(len(expectValueY)):
        yAvg += value * expectValueY[value]

    return expectXY - (xAvg * yAvg)


def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    result = 0

    for row in range(len(P)):
        for column in range(len(P[row])):
            result += P[row][column] * f(row, column)

    return result
