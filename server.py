from typing import Literal
import torch
import copy
from base_model import MNISTConvNet

#初始化
#共享模型参数


class Server(MNISTConvNet):
    
    def aggregate(self, parameters_list):
        param_list = []
        for i in range(len(parameters_list[0])):
            p_list = [p[i] for p in parameters_list]
            mean = torch.mean(torch.stack(p_list), dim=0)
            param_list.append(mean)

        self.replace_parameters(param_list)
        
    '''

    def FedAvg(self,w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w))
        return w_avg
    '''