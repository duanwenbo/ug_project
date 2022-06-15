import sys
"Please replace the path with the root path of the folder 'ug_project' in your computer!"
sys.path.append("/Users/duanwenbo/Documents/CodeSpace/UG_Project/ug_project")

import torch
import numpy as np
import copy
import torch.nn.functional as F
from Environment.Channel import SISO_Channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGD:
    "Multi-path searching + ranking"
    def __init__(self,channel, N_STEPS, N_POINTS, LEARNING_RATE, SEED):
        self.channel = channel 
        self.K = channel.pair_num
        self.N_STEPS = N_STEPS
        self.N_POINTS = N_POINTS
        self.LEARNING_RATE = LEARNING_RATE
        self.SEED = SEED
    
    def _generate_obj_func(self,channel, P, P_max = torch.tensor(1), var_noise = torch.tensor(1)):
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
        
    def _sum_rate(self, H, p, Pmax=1, var_noise=1):
        n = H.shape[0]
        rate = 0
        for i in range(n):
            S = p[i] * Pmax * (H[i, i] ** 2)
            t = H[i, :]
            I = var_noise + np.sum(p * Pmax * (H[i, :] ** 2)) - S
            rate += np.log2(1 + S / I)
        return rate


    def single_point_search(self,init_position):
        def _over_flow(power):
            # check whether stop ascenting      
            if "-" in str(power):
                return True
            elif "1." in str(power):
                return True
            else:
                return False
        
        channel = np.reshape(self.channel.CSI, (self.K, self.K))
        # gradient ascent
        power = copy.deepcopy(init_position)
        last_power = copy.deepcopy(init_position)
        # transform the channel into differentiable  tensor format
        obj_func = self._generate_obj_func(channel, power)
        for i in range(self.N_STEPS):
            obj_func.backward(retain_graph=True)
            with torch.no_grad():
                power += self.LEARNING_RATE * power.grad

            if _over_flow(power):
                break
            last_power = copy.deepcopy(power)

        capacity = self._sum_rate(channel, last_power.detach().numpy(), var_noise=1)

        if str(last_power) == str(init_position):
            modulus = 0
        else:
            modulus = torch.linalg.norm(last_power.grad, 2)

        return {"initial_position":init_position.detach().numpy(),
                "end_position": last_power.detach().numpy(),
                "N_STEPS": i+1,
                "modulus": modulus,
                "performance": capacity}


    def solve(self):
        result = []
        # the gradient ascent search
        for _ in range(self.N_POINTS):
            init_position = torch.rand(self.K, dtype=torch.float32, requires_grad=True)
            search_result = self.single_point_search(init_position)
            result.append((search_result['performance'],search_result['end_position']))
    
        best_result = sorted(result)[-1]
        best_capacity, best_power_ratio = best_result[0],list(best_result[1])
        return {"best_capacity":best_capacity,
                "best_power_ratio":best_power_ratio}

if __name__ == "__main__":
    channel = SISO_Channel(N_pairs=4,
                           seed=308)
    sgd = SGD(channel=channel,
                 N_STEPS=50,
                 N_POINTS=100,
                 LEARNING_RATE=0.001,
                 SEED=308)
    result = sgd.solve()
    print(result)