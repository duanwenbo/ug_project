import numpy as np
import math

# main functon of WMMES algorithm
# Adapted from https://github.com/Haoran-S/SPAWC2017
def best_ratio(p_int, H, Pmax=1, var_noise=1):
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


def sum_rate( H, p, Pmax=1, var_noise=1):
    n = H.shape[0]
    rate = 0
    for i in range(n):
        S = p[i] * Pmax * (H[i, i] ** 2)
        t = H[i, :]
        I = var_noise + np.sum(p * Pmax * (H[i, :] ** 2)) - S
        rate += np.log2(1 + S / I)
    return rate

def WMMES(channel,N_PAIRS):
    p_init = np.ones(N_PAIRS)
    power_allocation = best_ratio(p_init, channel)
    capacity = sum_rate(channel, power_allocation)
    return {"best_capacity":capacity,
            "best_power_ratio":power_allocation}


if __name__ == "__main__":
    pass
