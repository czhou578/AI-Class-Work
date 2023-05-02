'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
laplace_param = 0.0001


def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    results = []

    tag_count = Counter()
    word_tag_count = {}

    for sentence in train:
        for word, tag in sentence:
            if (word not in word_tag_count):
                word_tag_count[word] = Counter()
            tag_count[tag] += 1
            word_tag_count[word][tag] += 1

    max_tag = max(tag_count.keys(), key=(lambda key: tag_count[key]))

    for sentence in test:
        predicted = []
        for word_entry in sentence:
            if (word_entry not in word_tag_count):
                predicted.append((word_entry, max_tag))
            else:
                max_word_and_tag = max((word_tag_count[word_entry]).keys(
                ), key=lambda key: word_tag_count[word_entry][key])
                predicted.append((word_entry, max_word_and_tag))

        results.append(predicted)
    return results


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    laplace_param = 0.0001
    tags = set()
    words = set()
    initial_prob = {}

    for sentence in train:
        for pair in sentence:
            words.add(pair[0])
            tags.add(pair[1])

    list_tags = list(tags)
    list_words = list(words)

    # initial probability
    starting_tag_counter = Counter()
    for sentence in train:
        starting_tag_counter[sentence[0][1]] += 1

    for tag, count in starting_tag_counter.items():
        initial_prob[tag] = log(
            (laplace_param + count) / (len(train) + laplace_param * len(tags)))

    # transition probability

    transition_tag_pair_counter = Counter()
    transition_probability = {}

    for sentence in train:
        for idx in range(1, len(sentence)):
            current_tag = sentence[idx][1]
            previous_tag = sentence[idx - 1][1]
            transition_tag_pair_counter[(current_tag, previous_tag)] += 1

    transition_probability = transition_tag_pair_counter

    for tag1 in tags:
        denominator = 0
        for tag2 in tags:
            if ((tag1, tag2) in transition_tag_pair_counter):
                denominator += transition_tag_pair_counter[(tag1, tag2)]

        for tag in tags:
            if ((tag1, tag) in transition_tag_pair_counter):
                transition_probability[(tag1, tag)] = log((transition_probability[(
                    tag1, tag)] + laplace_param) / (denominator + laplace_param * len(tags)))
            else:
                transition_probability[(tag1, tag)] = log(
                    laplace_param / (len(train) + laplace_param * len(tags)))

    # emission probability

    emission_probability = Counter()
    tag_count = Counter()

    for sentence in train:
        for pair in sentence:
            tag_count[pair[1]] += 1
            emission_probability[pair] += 1

    for word in words:
        for tag in tags:
            if ((word, tag) in emission_probability):
                emission_probability[(word, tag)] = log((emission_probability[(
                    word, tag)] + laplace_param)/(tag_count[tag] + laplace_param * (len(words) + 1)))
            else:
                emission_probability[(word, tag)] = log(
                    laplace_param / (len(train) + laplace_param * (len(words) + 1)))

    trellis = [{}]

    '''
    for t in range(1, len(observations)):
        trellis.append({})
        for state in states:
            max_prob = max(trellis[t-1][prev_state]['prob'] * trans_prob[prev_state][state] * emit_prob[state][observations[t]] for prev_state in states)
            for prev_state in states:
                if trellis[t-1][prev_state]['prob'] * trans_prob[prev_state][state] * emit_prob[state][observations[t]] == max_prob:
                    trellis[t][state] = {'prob': max_prob, 'prev': prev_state}
                    break
    '''

    for tag in list_tags:
        trellis[0][tag] = {"probab": initial_prob['START'] *
                           emission_probability[(list_words[0], tag)], 'previous': None}

    # print(list_tags)

    for word_idx in range(1, len(list_words)):
        trellis.append({})
        for tag in list_tags:
            max_prob = max(trellis[word_idx - 1][prev_tag]['probab'] * transition_probability[(
                prev_tag, tag)] * emission_probability[(word, tag)] for prev_tag in list_tags if prev_tag in trellis[word_idx - 1].keys())

            for previous_tag in tags:
                if (previous_tag in trellis[word_idx - 1].keys()):
                    if trellis[word_idx - 1][previous_tag]['probab'] * transition_probability[(previous_tag, tag)] * emission_probability[(list_words[word_idx], tag)] == max_prob:
                        trellis[word_idx][tag] = {
                            'probab': max_prob, 'previous': previous_tag}
                        break

    print(trellis)

    sequence = []

    for sentence in test:
        temp_sentence = []
        for word in sentence:
            if (len(temp_sentence) == 0):
                temp_sentence.append((word, 'START'))
                continue

            word_index = list_words.index(word)
            max_prob = max(trellis[word_index][tag]['probab']
                           for tag in list_tags)
            current_state = max_prob

            for tag_idx in range(len(trellis) - 1, len(sentence) - 1, - 1):
                current_state = trellis[tag_idx][current_state]['previous']
                temp_sentence.append((word, trellis[tag_idx]))
            sequence.append(temp_sentence)
    print(sequence)

    '''
    for tag in list_tags:
        if (trellis[-1][tag]['probab'] == max_prob):
            current_state = tag
            break

    sequence.append(current_state)

    for tag_idx in range(len(trellis) - 1, 0, - 1):
        current_state = trellis[tag_idx][current_state]['previous']
        sequence.insert(0, current_state)
    '''
    # return sequence


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")
