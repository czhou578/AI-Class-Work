'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import random
import numpy as np
import torch
import torch.nn as nn

class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        
        '''
        #nested dict for Q table and N table
        convert the state and action into strings to be used to index Q['3]['1'] or N['1']['0']
        '''

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst

        qTable = np.zeros((state_cardinality[0], state_cardinality[1],state_cardinality[2],state_cardinality[3], state_cardinality[4], 3), dtype=float)
        nTable = np.zeros((state_cardinality[0], state_cardinality[1],state_cardinality[2],state_cardinality[3], state_cardinality[4], 3), dtype=float)
        
        self.qTable = qTable
        self.NTable = nTable

    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        negative_one_action = self.NTable[state[0], state[1], state[2], state[3], state[4], 0]
        zero_action = self.NTable[state[0], state[1], state[2], state[3], state[4], 1]
        one_action = self.NTable[state[0], state[1], state[2], state[3], state[4], 2]

        return [negative_one_action, zero_action, one_action]

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        explored_count = self.report_exploration_counts(state)

        if (all(action >= self.nfirst for action in explored_count)):
          return None

        while(True):
          random_num = random.randint(-1, 1)
          if explored_count[random_num] < self.nfirst:
            self.NTable[state[0], state[1], state[2], state[3], state[4], random_num] += 1
            return random_num

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''

        negative_one_action = self.qTable[state[0], state[1], state[2], state[3], state[4], 0]
        zero_action = self.qTable[state[0], state[1], state[2], state[3], state[4], 1]
        one_action = self.qTable[state[0], state[1], state[2], state[3], state[4], 2]

        return [negative_one_action, zero_action, one_action]

        # return self.qTable[state[0], state[1], state[2], state[3], state[4]]

    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q

        {
          key: {
            1: 2,

            2: 4,
            5: 1
          }
        }
        '''

        # print(newstate)
        # print(actions_of_state) #always zero
        # actions_of_state = self.qTable[newstate[0], newstate[1], newstate[2], newstate[3], newstate[4]]

        q_local = reward + self.gamma * max(self.report_q(newstate))

        return q_local
        # print(np.max(actions_of_state))
        # if np.max(actions_of_state) != 0.0:
        # print("from q_local", np.max(actions_of_state))
        
    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''

        q_local = self.q_local(reward, newstate)
        # print(action + 1
        #)
        print(action)
        if action != None:
          q_state = self.qTable[state[0], state[1], state[2], state[3], state[4], action+1]
          self.qTable[state[0], state[1], state[2], state[3], state[4], action+1] += self.alpha * (q_local - q_state)

        # if action == -1:
        #   # state_action_entry = self.qTable[state[0], state[1], state[2], state[3], state[4], 0]
        #   self.qTable[state[0], state[1], state[2], state[3], state[4], 0] += self.alpha * (self.q_local(reward, newstate) - self.qTable[state[0], state[1], state[2], state[3], state[4], 0])
        #   # print(state_action_entry)

        # elif action == 0:
        #   # state_action_entry = self.qTable[state[0], state[1], state[2], state[3], state[4], 1]
        #   self.qTable[state[0], state[1], state[2], state[3], state[4], 1] += self.alpha * (self.q_local(reward, newstate) - self.qTable[state[0], state[1], state[2], state[3], state[4], 1])

        # elif action == 1:
        #   # state_action_entry = self.qTable[state[0], state[1], state[2], state[3], state[4], 2]
        #   # print("self_local", self.q_local(reward, newstate))
        #   self.qTable[state[0], state[1], state[2], state[3], state[4], 2] += self.alpha * (self.q_local(reward, newstate) - self.qTable[state[0], state[1], state[2], state[3], state[4], 2])

    
    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        q = self.qTable
        n = self.NTable

        np.savez('trained_model.npz', q=q, n=n)

        
    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        data = np.load('trained_model.npz', allow_pickle=True)
        qTable = data['q']
        nTable = data['n']

    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''

        actions = self.qTable[state[0], state[1], state[2], state[3], state[4]]
        highest_q_value = max(actions)
        max_key = actions.argmax(int(highest_q_value))

        if max_key == 0:
          return (-1, highest_q_value)
        elif max_key == 1:
          return (0, highest_q_value)
        else:
          return (1, highest_q_value)
    
    def act(self, state):
        '''
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''

        actions = self.qTable[state[0], state[1], state[2], state[3], state[4]]
        fewer_then_nfirst = []
        
        for value in actions:
          if value < self.nfirst:
            fewer_then_nfirst.append(value)         

        if len(fewer_then_nfirst) > 0:
          action = self.choose_unexplored_action(state)
          return action
        else:
            random_num = np.random.random()
            if random_num < self.epsilon:
              random_number = random.randint(0, 2) - 1
              return random_number
            else:
              return self.exploit(state)[0]


class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        raise RuntimeError('You need to write this!')

    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.
        
        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        raise RuntimeError('You need to write this!')
        
    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.
        
        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state
        
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def load(self, filename):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
