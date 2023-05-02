'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3


def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    '''
    0 (left), 1 (up), 2 (right), or 3 (down)
    '''
    M = model.M
    N = model.N

    arr = np.zeros(M * N * 4 * M * N)

    arr = arr.reshape((M, N, 4, M, N))

    # up, down, left, right
    topBound = 0
    bottomBound = M - 1
    leftBound = 0
    rightBound = N - 1

    for m in range(M):
        for n in range(N):
            for idx in range(4):  # for each possible action
                neighborCellProb = []

                if model.T[m, n] == True:
                    arr[m, n, :, :, :] = 0
                    continue

                if idx == 0:  # going left

                    if m - 1 >= 0 and model.W[m - 1, n] == False:
                        arr[m, n, idx, m - 1, n] = model.D[m, n, 2]
                        neighborCellProb.append(model.D[m, n, 2])

                    if m + 1 < M and model.W[m + 1, n] == False:
                        arr[m, n, idx, m + 1, n] = model.D[m, n, 1]
                        neighborCellProb.append(model.D[m, n, 1])

                    if n - 1 >= 0 and model.W[m, n - 1] == False:
                        arr[m, n, idx, m, n - 1] = model.D[m, n, 0]
                        neighborCellProb.append(model.D[m, n, 0])
                    arr[m, n, idx, m, n] = 1 - sum(neighborCellProb)

                elif idx == 1:  # going up

                    if n - 1 >= 0 and model.W[m, n - 1] == False:
                        arr[m, n, idx, m, n - 1] = model.D[m, n, 1]
                        neighborCellProb.append(model.D[m, n, 1])

                    if n + 1 < N and model.W[m, n + 1] == False:
                        arr[m, n, idx, m, n + 1] = model.D[m, n, 2]
                        neighborCellProb.append(model.D[m, n, 2])

                    if m - 1 >= 0 and model.W[m - 1, n] == False:
                        arr[m, n, idx, m - 1, n] = model.D[m, n, 0]
                        neighborCellProb.append(model.D[m, n, 0])

                    arr[m, n, idx, m, n] = 1 - sum(neighborCellProb)

                elif idx == 2:  # going right

                    if m - 1 >= 0 and model.W[m - 1, n] == False:
                        arr[m, n, idx, m - 1, n] = model.D[m, n, 1]
                        neighborCellProb.append(model.D[m, n, 1])

                    if m + 1 < M and model.W[m + 1, n] == False:
                        arr[m, n, idx, m + 1, n] = model.D[m, n, 2]
                        neighborCellProb.append(model.D[m, n, 2])

                    if n + 1 < N and model.W[m, n + 1] == False:
                        arr[m, n, idx, m, n + 1] = model.D[m, n, 0]
                        neighborCellProb.append(model.D[m, n, 0])

                    arr[m, n, idx, m, n] = 1 - sum(neighborCellProb)

                elif idx == 3:  # going down

                    if m + 1 < M and model.W[m + 1, n] == False:
                        arr[m, n, idx, m + 1, n] = model.D[m, n, 0]
                        neighborCellProb.append(model.D[m, n, 0])

                    if n + 1 < N and model.W[m, n + 1] == False:
                        arr[m, n, idx, m, n + 1] = model.D[m, n, 1]
                        neighborCellProb.append(model.D[m, n, 1])

                    if n - 1 >= 0 and model.W[m, n - 1] == False:
                        arr[m, n, idx, m, n - 1] = model.D[m, n, 2]
                        neighborCellProb.append(model.D[m, n, 2])

                    arr[m, n, idx, m, n] = 1 - sum(neighborCellProb)

    return arr


def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    #
    shape_of_U = U_current.shape
    U_next = np.zeros((shape_of_U[0], shape_of_U[1]), dtype=float)

    '''
        R(S) - rewards array
        gamma - model.gamma
        U(s) - U_current(m, n)
        P(s'|s, a) = (m, n) and any of (0, 1, 2, 3) actions, p(m, n, a, m', n') where (m', n') is new state
        s' is all possible reachable states from s
    '''

    for m_idx in range(model.M):
        for n_idx in range(model.N):
            curr_reward = model.R[m_idx, n_idx]
            gamma = model.gamma
            curr_util = U_current[m_idx, n_idx]
            neighb_prob = []

            for idx in range(4):
                max_prob_utility = 0

                indices = np.nonzero(P[m_idx, n_idx, idx, :, :])
                row_cord = indices[0]
                col_cord = indices[1]

                for num_cord_idx in range(len(row_cord)):
                    m, n = row_cord[num_cord_idx], col_cord[num_cord_idx]
                    total = P[m_idx, n_idx, idx, m, n] * U_current[m, n]
                    max_prob_utility += total
                neighb_prob.append(max_prob_utility)
            U_next[m_idx, n_idx] = curr_reward + gamma * max(neighb_prob)

    return U_next


def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition_matrix(model)
    final_U = update_utility(model, P, np.zeros(
        (model.M, model.N), dtype=float))  # starting

    for i in range(100):
        temp_U = update_utility(model, P, final_U)
        epsilon_check = temp_U - final_U
        if np.any(abs(epsilon_check) >= epsilon):
            final_U = temp_U
        else:
            return temp_U

    return final_U


if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
