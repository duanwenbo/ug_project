"""
Author: Wenbo Duan
Date: 29th, April, 2022
Usage: Author: Wenbo Duan
Date: 29th, April, 2022
Usage: Main scripts for the algorithm: Gradient Descent with Random Restart (GDR)
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import sum_rate, WMMSE_sum_rate, generate_channel, timing
from gradient_ascent_V1 import SGDR
import wandb

# wandb.init(project="4x4")

P_MAX = 1
N_PAIRS = 4
SEED = 308
SNR_MAX = 10
SNR_STEP = 0.5
np.random.seed(SEED)


def random_dis(channel, SNR_ave, channel_name):
    # random allocating power
    print("processing random_dis with SNR {}...".format(SNR_ave))
    
    power_allocation = np.random.rand(N_PAIRS)
    result = sum_rate(channel, power_allocation, P_MAX, SNR_ave)
    return {"type":"Random", "SNR": 1/SNR_ave, "sum_rate":result, "K":N_PAIRS, 'channel_name':channel_name}


def uniform_dis(channel, SNR_ave, channel_name):
    print("processing uniform_dis with SNR {}...".format(SNR_ave))
    power_allocation = [1/N_PAIRS]*N_PAIRS
    result = sum_rate(channel, power_allocation, P_MAX, SNR_ave)
    return {"type":"Uniform", "SNR":  1/SNR_ave, "sum_rate":result, "K":N_PAIRS, 'channel_name':channel_name}


def WMMES(channel, SNR_ave, channel_name):
    print("processing WMMES with SNR {}...".format(SNR_ave))
    p_init = np.ones(N_PAIRS)
    power_allocation = WMMSE_sum_rate(p_init, channel)
    result = sum_rate(channel, power_allocation, P_MAX, SNR_ave)
    return {"type":"WMMES", "SNR": 1/SNR_ave, "sum_rate":result, "K":N_PAIRS, 'channel_name':channel_name}

# TODO: delete
def RL(channel, SNR_ave, channel_name, i):
    print("processing RL with SNR {}...".format(SNR_ave))
    choices = [0.8,1.6,0.6,1.3]
    return {"type":"VPG", "SNR": 1/SNR_ave, "sum_rate":choices[i], "K":N_PAIRS, 'channel_name':channel_name}



def gradient_descent(channel, SNR_ave, N_TRIALS = 50):
    print("processing gradient_descent with SNR {}...".format(SNR_ave))
    p_init = np.ones(N_PAIRS)
    result = SGDR(channel, SNR_ave, N_TRIALS, N_PAIRS)
    return {"type":"GDR", "SNR": 1/SNR_ave, "sum_rate":result, "K":N_PAIRS}

def experiment():
    global N_PAIRS
    result_list = []
    for k in [4]:
        N_PAIRS = k
        channel = generate_channel(K=N_PAIRS)
        channel = np.reshape(channel, (N_PAIRS,N_PAIRS))
        for SNR_ave in np.arange(0.1,SNR_MAX,SNR_STEP):
            result_list.append(uniform_dis(channel, SNR_ave))
            result_list.append(random_dis(channel, SNR_ave))
            result_list.append(WMMES(channel, SNR_ave))
            result_list.append(gradient_descent(channel, SNR_ave))
            print("\n")
        
    result_tabel = pd.DataFrame(result_list)
    result_tabel.to_csv("0425-1-cdf_plot.csv")
    return result_tabel

def experiment_02():
    # different channel realization
    # randomly drawn network with equal size
    data_points = 4
    Name = ['A','B','C','D']
    SNR_ave = 1

    result_list = []
    for i in range(data_points):
        print('performing channel {}'.format(i))
        channel_info = generate_channel(K=N_PAIRS)
        channel = np.reshape(channel_info, (N_PAIRS,N_PAIRS))
       
        result_list.append(uniform_dis(channel, SNR_ave, Name[i]))
        result_list.append(random_dis(channel, SNR_ave,Name[i]))
        result_list.append(WMMES(channel, SNR_ave, Name[i]))
        result_list.append(RL(channel,SNR_ave, Name[i], i))

        # result_list.append(gradient_descent(channel, SNR_ave, Name[i]))
        print(channel)
        
        
    result_tabel = pd.DataFrame(result_list)
    result_tabel.to_csv("0425-1-cdf_channel_different.csv")
    print(result_tabel)
    return result_tabel

def experiment_03():
    # different Kscal
    # scalability
    global N_PAIRS
    SNR_ave = 1
    REPEAT_TIME = 5

    result_list = []
    for _ in range(REPEAT_TIME):
        for i in [2,4,6,8,10]:
            N_PAIRS = i
            print('performing channel {}'.format(i))
            channel_info = generate_channel(K=N_PAIRS)
            channel = np.reshape(channel_info, (N_PAIRS,N_PAIRS))
        
            result_list.append(uniform_dis(channel, SNR_ave))
            result_list.append(random_dis(channel, SNR_ave))
            result_list.append(WMMES(channel, SNR_ave))
            # result_list.append(gradient_descent(channel, SNR_ave))
        
     
    result_tabel = pd.DataFrame(result_list)
    result_tabel.to_csv("0425-1-scale.csv")
    print(result_tabel)
    return result_tabel


def draw_graph():
    source =  pd.read_csv('0425-1-cdf_plot.csv')
    sns.lineplot(data=source.query("K==4"), x='SNR', y='sum_rate', hue='type', style='type',markers=True, dashes=False).set(title='Channel Capcity under different SNR | 4x4 SISO')
    # sns.lineplot(data=source.query("K==20"), x='SNR', y='sum_rate', markers=True, style='type', dashes=False)
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.ylabel("Sum-rate    bits/sec")
    plt.savefig('demo.png')
    print("figure saved")

def draw_cdf():
    df = pd.read_csv('0425-1-cdf_channel_different.csv')
    # x = df.query("type=='WMMES'")
    # print(x)
    sns.ecdfplot(data=df, x='sum_rate', hue='type').set(title='Emperical CDF | 4x4 SISO')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.ylabel("cummulative probability")
    plt.xlabel('sum-rate (bit/sec)')
    plt.savefig('cdf_test.png')


def draw_bar():
    # df = pd.read_csv('0425-1-scale.csv')
    source = experiment_02()
    # x = df.query("type=='WMMES'")
    # print(x)
    sns.barplot(data=source, x='channel_name', y='sum_rate', hue='type').set(title='Channel Capacity across random 4x4 channel | SEED = 308')
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.ylabel("Sum-rate (bit/sec)")
    plt.xlabel('Channel Name')
    plt.savefig('barplot_2.png')






    

if __name__ == "__main__":
    # channel = generate_channel(K=4)
    # channel = np.reshape(channel, (4,4))
    # for _ in range(1500):
    #     # uni = uniform_dis(channel,1)['sum_rate']
    #     # random = random_dis(channel,1)['sum_rate']
    #     wmmes = WMMES(channel, 1)['sum_rate']
    #     print(wmmes)
    #     wandb.log({'4x4':wmmes})
    draw_bar()


    

    