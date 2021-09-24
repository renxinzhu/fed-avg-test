from torch.nn import parameter
from dataset import generate_random_dataloader,generate_iid_dataloaders
import torch
from torch import nn
from client import Client
from server import  Server

#参数
lr = 0.05
local_epochs = 1
rounds = 300
num_clients = 100

#实例化
dataloaders = generate_iid_dataloaders(num_clients)

clients = [Client(idx) for idx in dataloaders]

server = Server(generate_random_dataloader())

loss_fn = nn.CrossEntropyLoss()
#训练
for i in range(rounds):
    parameters = [server.parameters]
    param_list = []

    for j in range(num_clients):
        client = clients[j]
        client.local_train(loss_fn, parameters=parameters,
                           E =local_epochs, lr=lr)
        client_parameter = [client.parameters()]
        param_list.append(client_parameter)


server.aggregate(param_list)

acc = server.validate()


