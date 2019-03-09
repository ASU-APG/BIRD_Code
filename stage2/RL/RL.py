import numpy as np
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import RLFunctions as RL

class BlocksPlanning:
    def __init__(self, state):
        self.state = state               
        self.discs = len(self.state)

    def move_allowed(self, state):
        RL.getvalidmoves(state)

    def get_moved_state(self, move):
        if self.move_allowed(move):
            disc_to_move = min(self.discs_on_peg(move[0]))
        moved_state = list(self.state)
        moved_state[disc_to_move] = move[1]
        return tuple(moved_state)

# Generates the reward matrix for the Towers of Hanoi game as a Pandas DataFrame
def generate_reward_matrix(N,MAX_BLOCKS, MAX_TOWERS):      # N is the number of discs
    #print("=======generating reward matrix=========")
    #print("states")
    states = list(i for i in itertools.product(list(range(MAX_BLOCKS+1)), repeat=MAX_TOWERS) if sum(i)<(MAX_BLOCKS+1))
    #print(states)
    states = list(itertools.product(list(range(3)), repeat=N))
    #print(states)
    #print("moves")
    moves = list(itertools.permutations(list(range(3)), 2))
    #print(moves)
    R = pd.DataFrame(index=states, columns=states, data=-np.inf)
    print(R)
    for state in states:
        tower = BlocksPlanning(state=state)
        for move in moves:
                next_state = tower.get_moved_state(move)
                R[state][next_state] = 0
    final_state = tuple([2]*N)          
    R[final_state] += 100               # Add a reward for all moves leading to the final state
    return R.values

def learn_Q(R, gamma=0.8, alpha=1.0, N_episodes=1000):
    Q = np.zeros(R.shape)
    states=list(range(R.shape[0]))
    for n in range(N_episodes):
        Q_previous = Q
        state = np.random.choice(states)                # Randomly select initial state
        next_states = np.where(R[state,:] >= 0)[0]      # Generate a list of possible next states
        next_state = np.random.choice(next_states)      # Randomly select next state from the list of possible next states
        V = np.max(Q[next_state,:])                     # Maximum Q-value of the states accessible from the next state
        Q[state, next_state] = (1-alpha)*Q[state, next_state] + alpha*(R[state, next_state] + gamma*V)      
    if np.max(Q) > 0:
        Q /= np.max(Q)      # Normalize Q to its maximum value
    return Q

def get_policy(Q, R):
    ploicy = RL.moveeffects(Q,R)
    return policy

def play(policy):
    start_state = 0
    end_state = len(policy)-1
    state = start_state
    moves = 0
    while state != end_state:
        state = np.random.choice(policy[state])
        moves += 1
    return moves

def play_average(policy, play_times=100):
    moves = np.zeros(play_times)
    for n in range(play_times):
        moves[n] = play(policy)
    return np.mean(moves), np.std(moves)

def Q_performance(R, episodes, play_times=100):
    means = np.zeros(len(episodes))
    stds = np.zeros(len(episodes))
    for n, N_episodes in enumerate(episodes):
        Q = learn_Q(R, N_episodes = N_episodes)
        policy = get_policy(Q,R)
        means[n], stds[n] = play_average(policy, play_times)
    return means, stds

def Q_performance_average(R, episodes, learn_times = 100, play_times=100):
    means_times = np.zeros((learn_times, len(episodes)))
    stds_times = np.zeros((learn_times, len(episodes)))
    for n in range(learn_times):
        means_times[n,:], stds_times[n,:] = Q_performance(R, episodes, play_times=play_times)
    means_averaged = np.mean(means_times, axis = 0)
    stds_averaged = np.mean(stds_times, axis = 0)
    return means_averaged, stds_averaged

N = 5                                   
MAX_BLOCKS = 5
MAX_TOWERS = 5
R = generate_reward_matrix(N, MAX_BLOCKS, MAX_TOWERS)
episodes = [1, 10, 100, 200, 300, 1000, 2000, 3000, 6000, 10000]
means_averaged, stds_averaged = Q_performance_average(R, episodes, learn_times=10, play_times=10)