import os
import time
import random
import torch
import numpy as np
from model import NNet, MCTS
from game import Board, ACTION_SIZE, rotate, rotate_bundle, HISTORY_SIZE
import pickle

def count_data(mcts: MCTS):
    count=0
    for key in mcts.Ns:
        if mcts.Ns[key] > 50:
            count+=1
    return count

def self_play(mcts: MCTS, timeout=10, id=0):
    board = Board()
    for _ in range(4):
        if random.random() < 0.5:
            board.place(*board.randplace())
    for _ in range(40):
        action = mcts.best_move(board, timeout)
        c = board.color
        v = mcts.query_v(board, action)
        board.place(*board.int2move(action))
        if hasattr(board, "winner"):
            break
        print(f'id {id} step = ', _)
        print(str(board), f"Color: {'O.X'[c+1]}, Action: {board.int2move(action)}, Value: {v}")
        if v < -0.9999 or v > 0.9999:
            break
            

def get_data(mcts: MCTS):
    key_list = list(mcts.Ns.keys())
    weight_list = [mcts.Ns[key] for key in key_list]
    sum = np.sum(weight_list)
    weight_list = [w/sum for w in weight_list]
    key_choose = np.random.choice(range(len(key_list)), 20000, replace=True, p=weight_list)
    data = []
    
    for id in key_choose:
        key = key_list[id]
        board = Board.from_state(key)
        input = board.bundled_input(board.legal_moves_input())
        ps = np.zeros(ACTION_SIZE)
        action = mcts.best_move(board, timeout=0)
        ps[action] = 1.0
        vs = mcts.Qsa[key, action]
        for i in range(4):
            data.append((
                input,
                ps,
                float(vs)
                )
            )
            input = rotate_bundle(input, board.n)
            last = ps[-1]
            ps = np.array(rotate(ps[:-1], board.n).tolist() + [last])
    return data


def self_play_and_get_data(ver: int):
    mcts = MCTS(None)
    for id in range(3):
        if os.path.exists("data/cnn.pt"):
            nnet = NNet(0, 128, 256)
            saved_state = torch.load("data/cnn.pt", map_location='cpu', weights_only=True)
            nnet.load_state_dict(saved_state)
            mcts.nnet = nnet
        self_play(mcts, timeout=20, id=id)
        data = get_data(mcts)
        with open(f"data{ver}-{id}.pkl", "wb") as f:
            pickle.dump(data, f)

# main
if __name__ == "__main__":
    random.seed(time.time())
    ver = 6
    while True:
        self_play_and_get_data(ver)
        ver += 1
