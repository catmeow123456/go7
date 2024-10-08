import time
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from game import ACTION_SIZE
from model import NNet, device, map_location


def save_mcts(mcts, filename):
    with open(filename, "wb") as f:
        pickle.dump(
            {
                "Ps": mcts.Ps,
                "Ns": mcts.Ns,
                "Qsa": mcts.Qsa,
                "Nsa": mcts.Nsa,
            }, f)


def load_mcts(mcts, filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        mcts.Ps = data["Ps"]
        mcts.Ns = data["Ns"]
        mcts.Qsa = data["Qsa"]
        mcts.Nsa = data["Nsa"]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_network(dataset, epoch_num=1000, pi_only=False):
    if not os.path.exists("data/cnn.pt"):
        nnet = NNet(0.5, 128, 256).to(device)
        nnet.apply(weights_init)
        torch.save(nnet.state_dict(), "data/cnn.pt")
    else:
        saved_state = torch.load("data/cnn.pt", map_location=map_location, weights_only=False)
        nnet = NNet(0, 128, 256).to(device)
        nnet.load_state_dict(saved_state)

    data_input = torch.tensor(np.array([d[0] for d in dataset]), dtype=torch.float32).to(device)
    data_output1, data_output2 = \
        torch.tensor(np.array([d[1] for d in dataset]), dtype=torch.float32).to(device), \
        torch.tensor(np.array([d[2] for d in dataset]), dtype=torch.float32).to(device)
    data_output1 = data_output1.view(-1, ACTION_SIZE)
    data_output2 = data_output2.view(-1, 1)

    # train nnet with data
    optimizer = optim.Adam(nnet.parameters(), lr=0.0005, weight_decay=1e-4)
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        output1, output2 = nnet(data_input)
        if not pi_only:
            # 计算交叉熵
            loss1 = torch.mean(-data_output1 * output1)
            loss2 = nn.MSELoss()(output2, data_output2)
            # 分别训练 loss1 和 loss2
            loss = loss1 + loss2
        else:
            loss = torch.mean(-data_output1 * output1)
        loss.backward()
        optimizer.step()
        if not pi_only:
            print(f'Epoch {epoch}, Loss1: {loss1.item()}, Loss2: {loss2.item()}')
        else:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # save nnet
    torch.save(nnet.state_dict(), "data/cnn.pt")


ver = 3
id = 2
while True:
    flag = False
    while not os.path.exists(f"data{ver}-{id}.pkl"):
        if not flag:
            flag = True
            print(f"Waiting for data{ver}-{id}.pkl")
        time.sleep(10)
    lst = [(ver, id)]
    for i in range(max(1, ver-1), ver + 1):
        for j in range(3):
            if i == ver and j > id:
                continue
            lst.append((i, j))
    for i, j in lst:
        print(f"Training data{i}-{j}.pkl")
        with open(f"data{i}-{j}.pkl", "rb") as f:
            data = pickle.load(f)
        train_network(data[0:60000], epoch_num=10)
    id += 1
    if id == 3:
        id = 0
        ver += 1
