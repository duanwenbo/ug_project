import numpy as np
import torch
import math
import time
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cpu")
np.random.seed(308)

def generate_channel(K=3):
    channel =  abs(1/np.sqrt(2)*(np.random.randn\
        (1,K**2)+1j*np.random.randn(1,K**2)))
    return channel


def sum_rate(H, p, Pmax=1, var_noise=1):
    n = H.shape[0]
    rate = 0
    for i in range(n):
        S = p[i] * Pmax * (H[i, i] ** 2)
        t = H[i, :]
        I = var_noise + np.sum(p * Pmax * (H[i, :] ** 2)) - S
        rate += np.log2(1 + S / I)
    return rate

def generate_obj_func(channel, P, P_max = torch.tensor(1), var_noise = torch.tensor(1)):
    # K: number of pairs; P: power control ratio, variables list
    # channel state info 
    P = F.sigmoid(P)
    CSI = torch.from_numpy(channel)
    CSI = CSI.to(device)
    P = P.to(device)
    P_max = P_max.to(device)
    var_noise = var_noise.to(device)
    K = channel.shape[0]
    sum_rate = 0
    for i in range(K):
        S = P[i] * P_max * (CSI[i,i]**2)
        I = var_noise +  torch.sum( P * P_max * (CSI[i,:]**2) ).to(device) - S
        sum_rate += torch.log2(1+S/I)
    return sum_rate

# main functon of WMMES algorithm
# Adapted from https://github.com/Haoran-S/SPAWC2017
def WMMSE_sum_rate(p_int, H, Pmax=1, var_noise=1):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-5:
            break
            

    p_opt = np.square(b) / Pmax
    return p_opt


def record_result(func):
    def wrapper(*args):
        table = func(*args)
        table = table.sort_values("performance")
        table.to_csv("result.csv")
        print("result saved")
    return wrapper

def timing(func):
    def wrapper(*args):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        running_time = end_time - start_time
        result["running_time"] = running_time
        return result
    return wrapper

def plot_result(table:pd.DataFrame):
    pd = table.sort_values("performance")
    max_2 = table.iloc[[-1,-2],-1].to_numpy()

    

        


