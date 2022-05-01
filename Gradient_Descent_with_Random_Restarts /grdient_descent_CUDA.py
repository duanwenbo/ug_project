"""
Author: Wenbo Duan
Date: 29th, April, 2022
Usage: Main scripts for the algorithm GDR with CUDA acceleration
"""

from time import time
from numpy import average
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from utils import WMMSE_sum_rate, generate_obj_func, generate_channel, sum_rate, record_result, timing
from HCS_condition import should_stop


K = 4  # number of pairs
N_CHANNEL = 16 # number of test channel
N_POINTS = 100 # number of path searched in each channel  ----- how many: k-meas
N_STEPS = 30 # number of maximum gradient steps in each path of each channel
LEARNING_RATE = 0.0001
SEED = 308
ADAPTIVE_STEPS = False

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# @timing
def _single_search(init_position, channel):
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
    
    # perparing for gradient ascent
    optimizer = optim.SGD([power], lr = LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_STEPS/4)
    
    for i in range(N_STEPS):
        optimizer.zero_grad()
        obj_func = - generate_obj_func(channel, power)
        obj_func.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        
        # if _over_flow(power):  
        #     break
        # if torch.linalg.norm(power.grad, 2) < 0.05:
        #     print(torch.linalg.norm(power.grad, 2))
        #     break
        last_power = copy.deepcopy(power)    

    capacity = sum_rate(channel, last_power.detach().numpy())

    if str(last_power) == str(init_position):
        modulus = 0
    else:
        modulus = torch.linalg.norm(last_power.grad, 2)
    

    return {"initial_position":init_position.detach().numpy(),
            "end_position": last_power.detach().numpy(),
            "N_STEPS": i+1,
            "modulus": modulus,
            "performance": capacity}


# @timing
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
# def multi_points_search(channel) -> pd.DataFrame:
#     performance_table = {}
#     # the baseline search result, for comparison
#     performance_table[0] = baseline_search(channel)
#     # the gradient ascent search
#     for i in range(N_POINTS):
#         init_position = torch.rand(K, dtype=torch.float32, requires_grad=True)
#         search_result = _single_search(init_position, channel)
#         performance_table[i+1] = search_result

#     performance_table = pd.DataFrame(performance_table).T
#     table = performance_table.sort_values("performance")

#     baseline_capacity = table.iloc[-1, -2] 
#     baseline_running_time = table.iloc[-1,-1] * 100
#     best_search_capacity = table.iloc[-2, -2]
#     search_time = table.iloc[:-1, -1].sum() 
 
  
#     return baseline_capacity, best_search_capacity, baseline_running_time, search_time

def multi_points_search(channel) :
    performance_table = {}
    # the baseline search result, for comparison
    performance_table[0] = baseline_search(channel)
    # the gradient ascent search

    i = 0
    search_results = []

    if ADAPTIVE_STEPS:
        while True:
            init_position = torch.rand(K, dtype=torch.float32, requires_grad=True)
            search_result = _single_search(init_position, channel)
            performance_table[i+1] = search_result

            search_results.append(search_result["performance"])
            if should_stop(search_results,0.2, 0.2, 0.01):
                break
    else:
        for i in range(N_POINTS):
            init_position = torch.rand(K, dtype=torch.float32, requires_grad=True)
            search_result = _single_search(init_position, channel)
            performance_table[i+1] = search_result


    performance_table = pd.DataFrame(performance_table).T
    table = performance_table.sort_values("performance")

    # [0] is the baseline result; [1] is the best searched result
    baseline_capacity = table.iloc[-1, -2]
    baseline_running_time = table.iloc[-1,-1] * 100
    best_search_capacity = table.iloc[-2, -2]
    search_time = table.iloc[:-1, -1].sum()

    gradient = table.iloc[-2,-3]

    # record_result(table)
  
    return baseline_capacity, best_search_capacity, baseline_running_time, search_time


def experiement():
    np.random.seed(SEED)
    result = {}
    for i in range(N_CHANNEL):
        channel = generate_channel(K)
        channel = np.reshape(channel, (K,K))
        baseline, experiment, bs_time, s_time = multi_points_search(channel)
        difference = (baseline - experiment) / baseline
        result[i] = {"baseline_result":baseline,
                    "best_search_result": experiment,
                    "difference":difference,
                    "baseline_time*100": bs_time,
                    "search_time": s_time,
                    "channel": channel}


    df = pd.DataFrame(result).T
    df.index = [chr(ord('a') + x).upper() for x in df.index]

    # df.to_csv("{0}x{0}_time_{1}_points.csv".format(K, N_POINTS))
   
   
    if ADAPTIVE_STEPS:
        searching_paths = "Adaptive searching"
    else:
        searching_paths = N_POINTS

    """bar plot"""
    # capacity = df.iloc[:,[0,1]]
    # average_difference = str(round(df['difference'].mean(), 4))
    # capacity.plot.bar(title="{0}x{0} SISO channel sum rate; Searching paths:{1}\nAverage difference:{2}".format(K,searching_paths, average_difference))
    # plt.xlabel("Random Channel Matrix")
    # plt.ylabel("Sum_rate - bits/s")
    # plt.legend()
    # plt.savefig("{0}x{0}_{1}_capacity_{2}.png".format(K, N_POINTS, searching_paths))
    
    # timing = df.iloc[:,[-3, -2]]
    # timing.plot.bar(title="Running time comparision;  Searching paths:{}".format(searching_paths))
    # plt.xlabel("Random Channel Matrix")
    # plt.ylabel("Running time - s")
    # plt.legend()
    # plt.savefig("{0}x{0}_{1}_time_{2}.png".format(K, N_POINTS, searching_paths))

    """for poster"""
    average_difference = str(round(df['difference'].mean(), 4))
    df["baseline_result"].plot(marker="o", ms=8, label="baseline_result (WMMSE)")
    df["best_search_result"].plot(marker="d", ms=8, label ="best_seach_result ({} restarting points)".format(N_POINTS) )
    plt.legend()


    # sns.lineplot(data=df["best_seach_result"])

    plt.title("{0}x{0} SISO channel; Average difference: {1}".format(K, average_difference))
    plt.xlabel("Channel (Random Seed = 308)")
    plt.ylabel("Sum Rate  bits/s")
  
    plt.savefig("{0}x{0}.png".format(K))

def time_compare():
    global K
    record_time = pd.DataFrame(columns=['Channel', 'time'])
    for t in range(5):
        for i in range(80):
            running_time = {}
            K = i+1
            running_time["Channel"] = K
            running_time["time"] = experiement()
            record_time = record_time.append(running_time, ignore_index=True)
    
    df = pd.DataFrame(record_time)
    
    
    sns.lineplot(data=df, x="Channel", y="time")
    plt.savefig("333.png")


def optia_starting():
    global K, N_STEPS, LEARNING_RATE
    K = 4
    N_STEPS = 30
    LEARNING_RATE = 1e-5

    channel = generate_channel(K)
    channel = np.reshape(channel, (K,K))
    result = baseline_search(channel)
    optima_position = result['end_position']
    optima_sumrate = result['performance']
    print("optima position: {};  optima result:{}".format(optima_position, optima_sumrate))
    print('\n')
    
    optima_position = torch.tensor(optima_position, dtype=torch.float32, requires_grad=True)
    search_result = _single_search(optima_position, channel)
    end_position = search_result['end_position']
    end_capacity = search_result['performance']
    print("searched position: {};  searched result:{}".format(end_position, end_capacity))











if __name__ =="__main__":
   experiement()
    
    