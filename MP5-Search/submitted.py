# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Implement bfs function

    path_queue = queue.Queue()
    path_queue.put([maze.start])

    visitedSet = set()
    visitedSet.add(maze.start)

    result = []

    while not path_queue.empty():

        frontPath = path_queue.get()  # a list

        if (maze.waypoints[0] in frontPath):
            result.append(frontPath)
            break

        cellNeighbors = maze.neighbors(
            frontPath[len(frontPath) - 1][0], frontPath[len(frontPath) - 1][1])

        for cell in cellNeighbors:
            if (maze.navigable(cell[0], cell[1]) and cell not in visitedSet):
                newPath = frontPath.copy()
                newPath.append(cell)

                path_queue.put(newPath)
                visitedSet.add(cell)

    return result[0]


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Implement astar_single

    path_queue = queue.PriorityQueue()

    path_queue.put((0, [maze.start]))

    visitedSet = set()
    visitedSet.add(maze.start)  # first element is distance from start

    waypoint = maze.waypoints[0]
    result = []

    while not path_queue.empty():
        existing_path = path_queue.get()

        if waypoint in existing_path[1]:
            result.append(existing_path[1])
            break

        cellNeighbors = maze.neighbors(
            existing_path[1][len(existing_path[1]) - 1][0], existing_path[1][len(existing_path[1]) - 1][1])

        for cell in cellNeighbors:

            if (maze.navigable(cell[0], cell[1]) and cell not in visitedSet):

                updated_Path = existing_path[1].copy()  # copy array

                manhatt_dist = abs(cell[0] - waypoint[0]) + \
                    abs(cell[1] - waypoint[1])

                f = len(updated_Path) - 1 + manhatt_dist

                updated_Path.append(cell)

                path_queue.put((f, updated_Path))

                visitedSet.add(cell)

    return result[0]

# This function is for Extra Credits, please begin this part after finishing previous two functions


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
