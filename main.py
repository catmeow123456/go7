"""
不知道为什么会写成这副鬼样子，以后有时间了再改一改，可能可以删掉
"""

import os
import torch
import numpy as np
from nnet import map_location, device
from game import Board
from mcts import NNet, MCTS
TIME_LIMIT = 4


class Player:
    def __init__(self):
        saved_state = torch.load(os.path.join("data", "cnn.pt"), map_location=map_location)
        self.nnet = NNet(0, 128, 256).to(device)
        self.nnet.load_state_dict(saved_state)
        self.mcts: MCTS = MCTS(self.nnet)
        self.board: Board = Board()

    def run(self, action: tuple[int, int] = None) -> tuple[int, int]:
        if action is not None:
            self.board.place(*action)
        new_action = self.mcts.best_move(self.board, TIME_LIMIT)
        x, y = self.board.int2move(new_action)
        self.board.place(x, y)
        if x == -1 or y == -1:
            return None
        return x, y
