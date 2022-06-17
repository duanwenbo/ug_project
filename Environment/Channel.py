
import sys
"Please replace the path with the root path of the folder 'ug_project' in your computer!"
sys.path.append("/Users/duanwenbo/Documents/CodeSpace/UG_Project/ug_project")

from gym import Env
from gym.spaces import Box
import numpy as np
from Environment.utils import WMMES

def sum_rate(pair_num, state_info, Pmax=1, var_noise=1):
    state_info = state_info[:-1]
    power = state_info[pair_num**2:]
    assert len(power) == pair_num, "checN_pairs checN_pairs length"
    H = state_info[:pair_num**2].reshape(pair_num, pair_num)
    rate = 0 
    for i in range(pair_num):
        S = power[i] * Pmax * (H[i, i] ** 2)
        I = var_noise + np.sum(power * Pmax * (H[i, :] ** 2)) - S
        rate += np.log2(1 + S / I)
    return rate


class SISO_Channel(Env):
    """
    In a NxN SISO channel

    STATE_SPACE:
    len(STATE_SPACE) = N^2 + N + 1 ,
    N^2: the channel state information, i.i.d. ,
    N: the power control ratio of each Tx ,
    1: the index of the Tx ,

    ACTION_SPACE:
    a real number between [-1,1] ,
    it stands for increasing / decreasing the power ratio of a Tx by a number between [-1,1]
    """
    metadata = {"render.modes":["console"]}
    def __init__(self, N_pairs, seed = 100):
        super(SISO_Channel, self).__init__()
        # the random seed to generate channel matrix
        self.seed = seed
        np.random.seed(self.seed)
        # number of transmitter-receiver pairs
        self.pair_num = N_pairs
        # power control ratio (between [0,1] )
        self.action_space = Box(low=np.array([-1]), high=np.array([1]))
        # total length of the state space: CSI + power control of each Tx + the position index of Tx
        num_state = N_pairs**2 + N_pairs + 1 
        self.observation_space = Box(low=np.array([0]*num_state), high=np.array([1]*num_state))
        # state space = Channel state information + Power control ratio
        self.CSI = abs(1/np.sqrt(2)*(np.random.randn(1,N_pairs**2)+1j*np.random.randn(1,N_pairs**2)))
        self.power_ratio = np.random.rand(1,N_pairs)
        self.Tx_position = np.array([[1.]])
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, self.Tx_position)
        # the time of a single epoch
        self.time = 200
        self.baseline = WMMES(np.reshape(self.CSI, (self.pair_num, self.pair_num)),N_pairs)
    
    def step(self, Tx_No, action):
        """
        Only update one Tx per  time
        Tx_No: which transmiter to be optimized
        action: increase / decrease the power
        """
        # the postion of the traget Tx in the state info
        Tx_index = self.pair_num**2 + Tx_No
        self.time -= 1
        self.state[Tx_index] += action

        # if the power control ratio of the target Tx exceed 1:
        #   stop training
        #   negative rewards to correct this policy
        if self.state[Tx_index] > 1:
            done = True
            reward = - self.state[Tx_index]
        # if the power control ratio of the target Tx is below 0:
        #   stop training
        #   negative rewards to correct this policy (less serious)
        elif self.state[Tx_index] < 0:
            done = True
            reward = self.state[Tx_index] - 1
        # if the optimization steps exceed the limits:
        #   stop training
        #   sum_rate as the reward to guide to improve the policy
        elif self.time < 0:
            done = True
            reward = sum_rate(self.pair_num, self.state)
        # if the power control ratio of the target Tx is within the target range:
        #  keep training
        else:
            reward = sum_rate(self.pair_num, self.state)
            done = False

        # to solve the prolem when the trajectory is only 1 step length.
        # We assume one more step will not affect the result as the rewards have been adjusted
        if self.time == 99:
            done = False

        info = {}
        return self.state, reward, done, info
    
    def reset(self, N_pairs):
        # N_pairs: the position index of Tx
        # np.random.seed(self.seed)
        # self.power_ratio = np.random.rand(1,self.pair_num)
        self.power_ratio = np.array([0.5]*self.pair_num) # fixed initial power ratio
        self.Tx_position = np.array([N_pairs])
        self.state = np.append(self.CSI, self.power_ratio)
        self.state = np.append(self.state, self.Tx_position)
        self.time = 100
        return self.state
    
    def render(self):
        pass

 

if __name__ == "__main__":
    channel = SISO_Channel(N_pairs=4,
                        seed=308)
    print(channel.baseline)
    


        


        
        