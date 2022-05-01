"""
Author: Wenbo Duan
Date: 29th, April, 2022
Usage: Main scripts for the algorithm: Gradient Descent with Random Restart (GDR)
"""

import torch
import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from utils import WMMSE_sum_rate, generate_obj_func, generate_channel, sum_rate, record_result


K = 4 # number of pairs
N_CHANNEL = 8 # number of test channel
N_POINTS = 100 # number of path searched in each channel
N_STEPS = 50 # number of maximum gradient steps in each path of each channel
LEARNING_RATE = 0.001
SEED = 308


def _single_search(init_position, channel, SNR_ave=1):
    def _over_flow(power):
        # check whether stop ascenting      
        if "-" in str(power):
            return True
        elif "1." in str(power):
            return True
        else:
            return False
    
    # gradient ascent
    power = copy.deepcopy(init_position)
    last_power = copy.deepcopy(init_position)
    # transform the channel into differentiable  tensor format
    obj_func = generate_obj_func(channel, power)
    for i in range(N_STEPS):
        obj_func.backward(retain_graph=True)
        with torch.no_grad():
            power += LEARNING_RATE * power.grad

        if _over_flow(power):
            break
        last_power = copy.deepcopy(power)

    capacity = sum_rate(channel, last_power.detach().numpy(), var_noise=SNR_ave)

    if str(last_power) == str(init_position):
        modulus = 0
    else:
        modulus = torch.linalg.norm(last_power.grad, 2)


    return {"initial_position":init_position.detach().numpy(),
            "end_position": last_power.detach().numpy(),
            "N_STEPS": i+1,
            "modulus": modulus,
            "performance": capacity}

def baseline_search(channel):
     # baseline performance
    p_init = np.ones(K)
    baseline_power = WMMSE_sum_rate(p_init, channel)
    baseline_performance = sum_rate(channel, baseline_power)

    return {"initial_position":p_init,
        "end_position": baseline_power,
        "N_STEPS": 0,
        "modulus": 0,
        "performance": baseline_performance}

# @record_result
def multi_points_search(channel) -> pd.DataFrame:
    performance_table = {}
    # the baseline search result, for comparison
    performance_table[0] = baseline_search(channel)
    # the gradient ascent search
    for i in range(N_POINTS):
        init_position = torch.rand(K, dtype=torch.float32, requires_grad=True)
        search_result = _single_search(init_position, channel)
        performance_table[i+1] = search_result

    performance_table = pd.DataFrame(performance_table).T
    table = performance_table.sort_values("performance")

    # [0] is the baseline result; [1] is the best searched result
    maximun = table.iloc[[-1,-2],-1].to_numpy()
    print(maximun)
    return maximun[0], maximun[1]


def experiement():
    # np.random.seed(SEED)
    result = {}
    for i in range(N_CHANNEL):
        channel = generate_channel(K)
        channel = np.reshape(channel, (K,K))
        baseline, experiment = multi_points_search(channel)
        result[i] = {"baseline_result":baseline,
                    "best_seach_result": experiment,
                    "channel": channel}

    df = pd.DataFrame(result).T
    print(df)

    df.index = [chr(ord('a') + x).upper() for x in df.index]
    df["baseline_result"].plot(marker="o", ms=8, label="baseline_result (WMMSE)")
    df["best_seach_result"].plot(marker="d", ms=8, label ="best_seach_result (200 restarting points)" )
    plt.legend()


    # sns.lineplot(data=df["best_seach_result"])

    plt.title("{0}x{0} SISO channel;\n Average Difference: {1} ".format(K, N_POINTS))
    plt.xlabel("Channel")
    plt.ylabel("Sum_rate")
  
    plt.savefig("{0}x{0}.png".format(K))
    print(df)


def SGDR(channel, SNR_ave, N_TRIALS, N_PAIRS):
    performance_table = []
    # the gradient ascent search
    for _ in range(N_TRIALS):
        init_position = torch.rand(N_PAIRS, dtype=torch.float32, requires_grad=True)
        search_result = _single_search(init_position, channel, SNR_ave)
        performance_table.append(search_result['performance'])
    performance_table.sort()
    best_result = performance_table[-1]
    return best_result

    





if __name__ =="__main__":
   experiement()
    
    