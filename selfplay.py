import os
import math
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
from game import Board, ACTION_SIZE, rotate_bundle, rotate, flip_bundle, flip, PASS
import pickle
from utils import Args
from multiprocessing import Pipe, Process, Queue, set_start_method

"""
所有超参数都放这里
"""
defaultargs = Args()


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


def selfplayEpsisode(id, mcts: MCTS, debug: bool = False, pipe = None):
    mcts.pipe = pipe
    board = Board()
    data = []
    while not board.haswinner:
        prob = mcts.getActionProb(board)
        s = board.hashed_state
        action = np.random.choice(range(ACTION_SIZE), p=prob)
        c = board.color
        v = mcts.Qsa.get((s, action), None)
        if v is not None and v < -0.9:
            board.winner = -c
            break
        # if v is not None and v > -0.005 and v < 0.005:
        #     board.winner = 0
        #     break
        if debug:
            print(f"Color: {'O.X'[c+1]}, Action: {board.int2move(action)}, Value: {v}", flush=True)

        input = board.bundled_input()
        if v is not None and action != PASS:
            res = 0
            for act in range(ACTION_SIZE):
                res += mcts.Qsa.get((s, act), 0) * prob[act]
            if debug:
                print(f"Average Qsa = {res}", flush=True)
            for _ in range(4):
                data.append((input, prob, res))
                data.append((flip_bundle(input, board.n),
                                np.append(flip(prob[:-1], board.n), prob[-1]),
                                res))
                input = rotate_bundle(input, board.n)
                prob = np.append(rotate(prob[:-1], board.n), prob[-1])

        board.place(*board.int2move(action))
        if debug:
            print(board, flush=True)
    if mcts.pipe is not None:
        mcts.pipe.send(None)
    if debug:
        print("Winner = ", "O.X"[board.winner + 1], flush=True)
    with open(os.path.join("tmp", f"{id}.pkl"), "wb") as f:
        pickle.dump(data, f)


def predict_together(nnet, pipe_list):
    end_ids = set()
    while True:
        id_list = []
        input_list = []
        for id in range(len(pipe_list)):
            if id in end_ids:
                continue
            input = pipe_list[id].recv()
            if input is None:
                end_ids.add(id)
                print(f"Process {id} end", flush=True)
                continue
            id_list.append(id)
            input_list.append(input)
        if len(input_list) == 0:
            print("All process end", flush=True)
            return
        input_list = np.array(input_list)
        output = nnet.predict_multiple(input_list)
        for id, out in zip(id_list, output):
            pipe_list[id].send(out)

class Coach:
    path: str
    args: Args
    info: dict

    def __init__(self, infopath: str, args: Args = None):
        self.path = infopath
        self.args = args if args is not None else defaultargs
        with open(infopath, "r") as f:
            self.info = json.load(f)

    def train(self, nnet: NNet, dataset: list) -> NNet:
        if nnet is None:
            nnet = NNet(0.5, 128, 256).to(device)
            nnet.apply(weights_init)
        batch_size = 48000
        epoch = math.ceil(len(dataset) / batch_size)
        for i in range(epoch):
            print(f"Epoch {i}, batch size {batch_size}", flush=True)
            data = dataset[i * batch_size: (i + 1) * batch_size]

            data_input = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(device)
            data_output1, data_output2 = \
                torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(device), \
                torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(device)
            data_output1 = data_output1.view(-1, ACTION_SIZE)
            data_output2 = data_output2.view(-1, 1)

            # train nnet with data
            optimizer = optim.Adam(nnet.parameters(), lr=0.0003, weight_decay=1e-4)
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
                print(f'Loss1: {loss1.item()}, Loss2: {loss2.item()}', flush=True)

        torch.cuda.empty_cache()
        return nnet

    def learn(self) -> NNet:
        nnet = None
        if "best" in self.info:
            nnet = load_model(self.info["best"])

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        num_workers = 14
        for _ in tqdm(range(self.args.numEps // num_workers)):
            mcts = MCTS(nnet, self.args)
            pipe_list = [Pipe() for _ in range(num_workers)]
            process_list = []
            for id in range(num_workers):
                process_list.append(Process(target=selfplayEpsisode,
                                            args=(id, mcts,
                                                  True if id == 0 else False, pipe_list[id][0])))
            process_aux = Process(target=predict_together,
                                  args=(nnet, [pipe[1] for pipe in pipe_list]))
            for process in process_list:
                process.start()
            process_aux.start()
            for process in process_list:
                process.join()
            process_aux.join()
            print('All process joined')
            for pipe in pipe_list:
                pipe[0].close()
                pipe[1].close()
            for process in process_list:
                process.terminate()
            process_aux.terminate()
            for id in range(num_workers):
                with open(os.path.join("tmp", f"{id}.pkl"), "rb") as f:
                    data = pickle.load(f)
                trainExamples.extend(data)
            # tqdm.write(str(len(trainExamples)))
            if len(trainExamples) >= self.args.maxlenOfQueue:
                break
        random.shuffle(trainExamples)
        id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join("data", f"dataset-{id}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(trainExamples, f)
        self.info["dataset"].append(filename)
        while len(self.info["dataset"]) > self.args.numItersForTrainExamplesHistory:
            self.info["dataset"].pop(0)
        with open(self.path, "w") as f:
            json.dump(self.info, f)

        trainingData = []
        for filename in self.info["dataset"]:
            with open(filename, "rb") as f:
                trainingData.extend(pickle.load(f))
        random.shuffle(trainingData)
        newnnet = self.train(mcts.nnet, trainingData)
        filename = os.path.join("data", f"nnet-{id}.pt")
        torch.save(newnnet.state_dict(), filename)
        return newnnet, filename

    def compete(self, nnet1: NNet, nnet2: NNet) -> tuple:
        win, lose = 0, 0
        mcts1 = MCTS(nnet1, self.args)
        mcts2 = MCTS(nnet2, self.args)
        flag = 1
        for _ in range(self.args.arenaCompare):
            if flag == 1:
                role = {1: mcts1, -1: mcts2}
            else:
                role = {1: mcts2, -1: mcts1}
            board = Board()
            while not board.haswinner:
                action = role[board.color].nnet_move(board)
                board.place(*board.int2move(action))
            if board.winner == flag:
                win += 1
            elif board.winner == -flag:
                lose += 1
            print(f"Win = {win}, Lose = {lose}", flush=True)
            flag = -flag
        return win, lose

    def run(self):
        for _ in range(self.args.numIters):
            newnnet, filename = self.learn()
            win, lose = self.compete(newnnet, load_model(self.info["best"]))
            print(f"Win = {win}, Lose = {lose}", flush=True)
            if win / (win + lose) >= self.args.updateThreshold:
                print(f"Accept new model {filename}", flush=True)
                self.info["best"] = filename
                self.info["models"].append(filename)
                with open(self.path, "w") as f:
                    json.dump(self.info, f)
            else:
                print(f"Reject new model {filename}", flush=True)


if __name__ == '__main__':
    set_start_method('spawn')
    coach = Coach(os.path.join("data", "info.json"))
    coach.run()
