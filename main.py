import torch
import numpy as np
from game import Board
from model import NNet, MCTS
TIME_LIMIT = 4


class Player:
    def __init__(self):
        saved_state = torch.load("data/cnn.pt", map_location="cpu")
        self.nnet = NNet(0, 128, 256)
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