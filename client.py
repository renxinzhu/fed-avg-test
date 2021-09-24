from torch.nn.functional import cross_entropy
from torch import nn
import torch
from base_model import MNISTConvNet


class Client(MNISTConvNet):
    # E local epoch
    def local_train(self, loss_fn, lr: float, parameters=None, E = 100):
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        for _  in range(E):
            for X,y in self.dataloader:

                pred = self.forward(X)
                loss = loss_fn(pred,y)
                #一次梯度下降
                opt.zero_grad()
                loss.backward()
                opt.step()

                