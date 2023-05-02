from collections import Counter
import numpy as np

# test = [[("word1", "tag1"), ("word2", "tag2")], [
#     ("word3", "tag3"), ("word4", "tag4"), ("word4", "tag4")]]
# result = Counter()
# for entry in test:
#     result += Counter(entry)
#     # print(Counter(entry))
# print(result.most_common())
# print(result)
# word3

# tags = set()
# tags.add("hi")
# tags.add("bi")
# tags.add("ai")
# tags.add("ni")
# tags.add("zi")

# matrix = []
# for i in range(5):
#     matrix.append({tag: 0 for tag in tags})

# print(matrix)

# trellis = np.zeros((len(states), len(observations)))
# create a 2D array of tuples
arr = np.array([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10), (11, 12)]])

# select the first index of all the elements in the first column
first_col = arr[:, 0, 0]
print(first_col)

# select the first element of the first column
first_element = first_col[0]

print(first_element)
