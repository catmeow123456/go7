import copy
import random
import numpy as np
from referee import Referee
from typing import List
import itertools
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
N = 7
BLACK = 1
WHITE = -1
ACTION_SIZE = 50
PASS = 49
HISTORY_SIZE = 3


class Board:
    n: int = N
    board: np.ndarray
    referee: Referee
    color: int = BLACK

    def __init__(self) -> None:
        self.board = np.zeros((self.n, self.n), dtype=np.int32)
        self.referee = Referee(self.n)

    def from_state(buffer) -> 'Board':  # 仅在训练时用到
        previous_pass = False
        obj = object.__new__(Board)
        obj.n = N
        obj.board = np.frombuffer(buffer, dtype=np.int32).reshape((obj.n, obj.n))
        obj.color = BLACK
        obj.referee = Referee(obj.n)
        obj.referee.previous_pass = previous_pass
        return obj

    def move2int(self, x: int, y: int) -> int:
        return x * self.n + y

    def int2move(self, a: int):
        if a == PASS:
            return -1, -1
        return a // self.n, a % self.n

    def copy(self):
        return copy.deepcopy(self)

    def ask(self, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= self.n or y >= self.n:
            return -2
        return self.board[x][y]

    def hashed_state(self):
        return (self.board * self.color).astype(np.int32).tobytes()

    def place(self, x: int, y: int) -> bool:  # 如果第二次 PASS，返回 True
        if x == -1 and y == -1:
            newboard, winner = self.referee.judge(self.board.tolist(), None)
            if winner != 0:
                self.winner = winner
                self.color = -self.color
                return True
        else:
            newboard, winner = self.referee.judge(self.board.tolist(), (x, y))
            try:
                assert newboard is not None
            except AssertionError:
                print('debug!')
                print(self.__str__())
                print(x, y)
                import pickle
                with open('debug.pkl', 'wb') as f:
                    pickle.dump(self, f)
                raise Exception
        assert winner == 0
        self.board = np.array(newboard, dtype=np.int32)
        self.color = -self.color
        return False

    def randplace(self):
        moves = self.legal_moves()
        if len(moves) == 0:
            return -1, -1
        return random.choice(moves)

    def legal_moves(self) -> List:
        return self.referee.get_valid(self.board.tolist())

    def __str__(self) -> str:
        s = ""
        for i in range(self.n):
            for j in range(self.n):
                s += "O_X"[self.board[i][j] + 1]
            s += "\n"
        return s

    def bundled_input(self, valids: np.ndarray) -> np.ndarray:
        input = [[0] * (self.n * self.n) for _ in range(HISTORY_SIZE * 2 + 1)]
        for i in range(min(HISTORY_SIZE, len(self.referee.previous_states))):
            board = list(itertools.chain(*self.referee.previous_states[-1-i]))
            arr1 = list(map(lambda x: 1 if x == self.color else 0, board))
            arr2 = list(map(lambda x: 1 if x == -self.color else 0, board))
            input[i*2+1] = arr1
            input[i*2+2] = arr2
        arr3 = valids.tolist()[:-1]
        input[0] = arr3
        return np.array(input, dtype=np.float32)

    def legal_moves_input(self) -> np.ndarray:
        moves = self.legal_moves()
        valids = np.zeros(ACTION_SIZE)
        if moves:
            for move in moves:
                valids[self.move2int(*move)] = 1
        valids[PASS] = 1
        return valids


def rotate(board: np.ndarray, N: int) -> np.ndarray:
    return np.rot90(board.reshape(N, N)).flatten()


def rotate_bundle(bundle: np.ndarray, N: int) -> np.ndarray:
    return np.array([rotate(bundle[i], N) for i in range(len(bundle))])


def evaluate(board: Board) -> tuple[int, int]:
    score = {1: 0, -1: 0}
    n = board.n
    visit = np.zeros((n, n), dtype=np.bool_)
    szX = np.zeros((n, n), dtype=np.int32)
    szO = np.zeros((n, n), dtype=np.int32)
    queue = []

    def dfs(x, y, ac) -> int:
        if x < 0 or x >= n or y < 0 or y >= n:
            return 0
        if visit[x][y]:
            return 0
        if board.board[x][y] == ac:
            return 0
        visit[x][y] = True
        queue.append((x, y))
        return 1 + dfs(x-1, y, ac) + dfs(x+1, y, ac) + dfs(x, y-1, ac) + dfs(x, y+1, ac)

    for i in range(n):
        for j in range(n):
            c = board.board[i][j]
            if c != 1 and not visit[i][j]:
                sz = dfs(i, j, 1)
                for x, y in queue:
                    szX[x][y] = sz
                queue.clear()

    visit.fill(False)
    for i in range(n):
        for j in range(n):
            c = board.board[i][j]
            if c != -1 and not visit[i][j]:
                sz = dfs(i, j, -1)
                for x, y in queue:
                    szO[x][y] = sz
                queue.clear()

    sz = {1: szX, -1: szO}

    def calc_score(c: np.int32) -> int:
        if c < 6:
            return 1
        return int((6 + (c - 6) / 3) / c)

    for i in range(n):
        for j in range(n):
            c = board.board[i][j]
            if c != 0:
                if sz[-c][i][j] > 7:
                    score[c] += 1
            if c != 1:
                sz1 = sz[1][i][j]
                if sz1 <= 17:
                    score[1] += calc_score(sz1)
            if c != -1:
                sz2 = sz[-1][i][j]
                if sz2 <= 17:
                    score[-1] += calc_score(sz2)
    return score[1], score[-1]
