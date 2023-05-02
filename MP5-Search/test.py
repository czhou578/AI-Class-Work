import queue


def test():
    path_queue = queue.PriorityQueue()
    # path_queue.put((0, [(0, (1, 3))]))
    testing = (1, 2)
    print(testing[0])
    path_queue.put([(0, (1, 3)), (2, (1, 3))])
    path_queue.put([(0, (1, 3)), (1, (1, 5))])
    path_queue.put([(0, (1, 3)), (4, (1, 6))])
    path_queue.put([(0, (1, 3)), (5, (1, 1))])

    # first = path_queue.get()
    # array = first[0].copy()

    # array.append((1, (2, 3)))
    # first[0] = 3
    # first[1] = array

    # path_queue.put(first)

    # reversed_array = firstArray.reverse()
    # print('reversed', firstArray[::-1])
    # path_queue.put([(2, (10, 2)), (2, (1, 2)), (3, (1, 2))])

    print(path_queue.get())


test()
