import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
import numpy as np
from collections import deque
from nnet import device, map_location, NNet
from mcts import MCTS
from game import Board, ACTION_SIZE, rotate_bundle
import pickle
from utils import dotdict

"""
所有超参数都放这里
"""
defaultargs = dotdict({
    'numIters': 1000,  # 策略迭代 numIters 次数
    'numEps': 100,  # 进行完整的 numEps 轮游戏
    'numMCTSSims': 50,  # 在每一轮游戏的每一个节点，运行 numMCTSSims 次 MCTS 模拟后再采样行动和获取样本
    'maxlenOfQueue': 20000,  # maxlenOfQueue 个样本为一组
    'numItersForTrainExamplesHistory': 5,  # 保留最近 numItersForTrainExamplesHistory 组样本，将他们混合后 shuffle 出训练集
    'arenaCompare': 40,  # 与历史模型对弈 arenaCompare 次，用于评估新模型的优劣
    'updateThreshold': 0.55,  # 新模型胜率超过 updateThreshold 时，接受新模型
})


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_model(filename: str) -> NNet:
    nnet = NNet(0.5, 128, 256).to(device)
    saved_state = torch.load(filename, map_location=map_location, weights_only=True)
    nnet.load_state_dict(saved_state)
    return nnet


class Coach:
    path: str
    args: dotdict
    info: dotdict

    def __init__(self, infopath: str, args: dotdict = None):
        self.path = infopath
        self.args = args if args is not None else defaultargs
        with open(infopath, "r") as f:
            data = json.load(f)
            self.info = dotdict(data)

    def train(self, nnet: NNet, dataset: list) -> NNet:
        if nnet is None:
            nnet = NNet(0.5, 128, 256).to(device)
            nnet.apply(weights_init)
        batch_size = 64000
        epoch = len(dataset) // batch_size
        for i in range(epoch):
            print(f"Epoch {i}, batch size {batch_size}")
            data = dataset[i * batch_size: (i + 1) * batch_size]

            data_input = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(device)
            data_output1, data_output2 = \
                torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(device), \
                torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(device)
            data_output1 = data_output1.view(-1, ACTION_SIZE)
            data_output2 = data_output2.view(-1, 1)

            # train nnet with data
            optimizer = optim.Adam(nnet.parameters())  # , weight_decay=1e-4)
            for _ in range(10):
                output1, output2 = nnet(data_input)
                # 计算交叉熵
                loss1 = torch.mean(-data_output1 * output1)
                loss2 = nn.MSELoss()(output2, data_output2)
                # 分别训练 loss1 和 loss2
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Loss1: {loss1.item()}, Loss2: {loss2.item()}')

        return nnet

    def selfplayEpsisode(self, mcts: MCTS):
        board = Board()
        # for _ in range(4):
        #     if random.random() < 0.5:
        #         board.place(*board.randplace())
        data = []
        # print(board)
        while not board.haswinner:
            prob = mcts.getActionProb(board)
            action = np.random.choice(range(ACTION_SIZE), p=prob)
            c = board.color

            input = board.bundled_input()
            for _ in range(4):
                data.append((input, c, prob))
                input = rotate_bundle(input, board.n)

            board.place(*board.int2move(action))
            # v = mcts.query_v(board, action)
            # print(str(board), f"Color: {'O.X'[c+1]}, Action: {board.int2move(action)}, Value: {v}")
        data = [(d[0], d[2], 1 if d[1] == board.winner else -1) for d in data]
        return data

    def learn(self) -> NNet:
        mcts = MCTS(None, self.args)
        if hasattr(self.info, "best"):
            mcts.nnet = load_model(self.info.best)

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for _ in tqdm(range(self.args.numEps)):
            data = self.selfplayEpsisode(mcts)
            trainExamples.extend(data)
            tqdm.write(f"Collecting training data {len(trainExamples)}")
            if len(trainExamples) >= self.args.maxlenOfQueue:
                break
        random.shuffle(trainExamples)
        id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"data/dataset-{id}.pkl"
        with open(f"data/dataset-{id}.pkl", "wb") as f:
            pickle.dump(trainExamples, f)
        self.info.dataset.append(filename)
        if len(self.info.dataset) > self.args.numItersForTrainExamplesHistory:
            self.info.dataset.pop(0)

        trainingData = []
        for filename in self.info.dataset:
            with open(filename, "rb") as f:
                trainingData.extend(pickle.load(f))
        random.shuffle(trainingData)
        newnnet = self.train(mcts.nnet, trainingData)
        filename = f"data/nnet-{id}.pt"
        torch.save(newnnet.state_dict(), filename)
        return newnnet, filename

    def compete(self, nnet1: NNet, nnet2: NNet) -> tuple:
        win, lose = 0, 0
        for _ in range(self.args.arenaCompare):
            mcts1 = MCTS(nnet1, self.args)
            mcts2 = MCTS(nnet2, self.args)
            flag = 1
            if random.random() < 0.5:
                role = {1: mcts1, -1: mcts2}
            else:
                role = {1: mcts2, -1: mcts1}
                flag = -1
            board = Board()
            while not board.haswinner:
                prob = role[board.color].getActionProb(board, temp=0)
                action = np.random.choice(range(ACTION_SIZE), p=prob)
                board.place(*board.int2move(action))
            if board.winner == flag:
                win += 1
            else:
                lose += 1
        return win, lose

    def run(self):
        for _ in range(self.args.numIters):
            newnnet, filename = self.learn()
            win, lose = self.compete(newnnet, load_model(self.info.best))
            if win / (win + lose) > self.args.updateThreshold:
                print(f"Accept new model {filename}")
                self.info.best = filename
                self.info.models.append(filename)
                with open(self.path, "w") as f:
                    json.dump(self.info, f)
            else:
                print(f"Reject new model {filename}")


coach = Coach("data/info.json")
coach.run()
