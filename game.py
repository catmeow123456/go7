"""
这个文件应该是没有 bug 了！
"""

import copy
import random
import numpy as np
from referee import Referee
from typing import List
import itertools
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
N = 7
BLACK = np.int8(1)
WHITE = np.int8(-1)
ACTION_SIZE = 50
PASS = 49
HISTORY_SIZE = 3


class Board:
    n: int = N
    board: np.ndarray
    referee: Referee
    color: np.int8 = BLACK

    def __init__(self) -> None:
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.referee = Referee(self.n)

    def __str__(self) -> str:
        """
        用于打印棋盘状态。
        """
        s = ""
        for i in range(self.n):
            for j in range(self.n):
                s += "O.X"[self.board[i][j] + 1]
            s += "\n"
        return s

    def move2int(self, x: int, y: int) -> int:
        return x * self.n + y

    def int2move(self, a: int):
        if a == PASS:
            return -1, -1
        return a // self.n, a % self.n

    def copy(self):
        return copy.deepcopy(self)

    def from_state(buffer) -> 'Board':
        """
        从一个哈希值构造一个游戏状态。
        """
        obj = object.__new__(Board)
        obj.n = N
        arr = np.frombuffer(buffer, dtype=np.int8).reshape((-1, obj.n, obj.n))
        obj.board = arr[0]
        obj.color = BLACK
        obj.referee = Referee(obj.n)
        for i in range(HISTORY_SIZE):
            if arr[-1-i].any():
                obj.referee.previous_states.append(arr[-1-i].tolist())
        try:
            obj.referee.previous_pass = arr[0].any() and arr[0].tolist() == arr[1].tolist()
        except:
            print(arr, arr.__class__)
            raise Exception
        return obj

    @property
    def hashed_state(self):
        """
        返回一个游戏状态的哈希值
        """
        input = np.zeros((HISTORY_SIZE, self.n, self.n), dtype=np.int8)
        for i in range(min(HISTORY_SIZE, len(self.referee.previous_states))):
            board = np.array(self.referee.previous_states[-1-i], dtype=np.int8)
            input[i] = board * self.color
        input = input.flatten()
        arr = bytes(input)
        return arr

    @property
    def haswinner(self) -> bool:
        return hasattr(self, "winner")

    def place(self, x: int, y: int) -> bool:
        """
        在 (x,y) 处落子，返回是否成功。
        如果 (x,y) 为 (-1,-1)，则表示 pass。
        如果两次 pass, 则游戏结束，游戏决出胜负，可以通过 self.winner 获取胜者。
        """
        if hasattr(self, "winner"):
            return False
        if x == -1 and y == -1:
            newboard, winner = self.referee.judge(self.board.tolist(), None)
            if winner != 0:
                self.winner = winner
                self.color = -self.color
                return True
        else:
            newboard, winner = self.referee.judge(self.board.tolist(), (x, y))
            if newboard is None:
                return False
        assert winner == 0
        self.board = np.array(newboard, dtype=np.int8)
        self.color = -self.color
        return True

    def randplace(self):
        """
        返回一个随机的合法的落子位置，如果没有合法位置，则返回 (-1,-1)。
        """
        moves = self.legal_moves()
        if len(moves) == 0:
            return -1, -1
        return random.choice(moves)

    def legal_moves(self) -> List:
        """
        返回一个列表，包含所有合法的落子位置。
        """
        return self.referee.get_valid(self.board.tolist())

    def bundled_input(self, valids: np.ndarray = None) -> np.ndarray:
        """
        返回一个包含历史状态的输入，形状为 (HISTORY_SIZE*2+1, N*N)。
        该输入可以被包装为 tensor 用于神经网络的输入。
        """
        input = [[0] * (self.n * self.n) for _ in range(HISTORY_SIZE * 2 + 1)]
        for i in range(min(HISTORY_SIZE, len(self.referee.previous_states))):
            board = list(itertools.chain(*self.referee.previous_states[-1-i]))
            arr1 = list(map(lambda x: 1 if x == self.color else 0, board))
            arr2 = list(map(lambda x: 1 if x == -self.color else 0, board))
            input[i*2+1] = arr1
            input[i*2+2] = arr2
        if valids is None:
            valids = self.legal_moves_input()
        arr3 = valids.tolist()[:-1]
        input[0] = arr3
        return np.array(input, dtype=np.float32)

    def legal_moves_input(self) -> np.ndarray:
        """
        返回一个长度为 ACTION_SIZE 的 np.ndarray 类型的向量（其中最后一格表示动作 PASS），
        用于表示可行的动作空间，可以作为 bundled_input 的第一个 channel。
        """
        moves = self.legal_moves()
        valids = np.zeros(ACTION_SIZE)
        if moves:
            for move in moves:
                valids[self.move2int(*move)] = 1
        valids[PASS] = 1
        return valids


def rotate(board: np.ndarray, N: int) -> np.ndarray:
    """
    辅助函数，将棋盘逆时针旋转 90 度。
    """
    return np.rot90(board.reshape(N, N)).flatten()


def rotate_bundle(bundle: np.ndarray, N: int) -> np.ndarray:
    """
    辅助函数，将棋盘的 bundled_input 逆时针旋转 90 度。
    """
    return np.array([rotate(bundle[i], N) for i in range(len(bundle))])

def flip(board: np.ndarray, N: int) -> np.ndarray:
    """
    辅助函数，将棋盘水平翻转
    """
    return np.flip(board, axis=0)

def flip_bundle(bundle: np.ndarray, N: int) -> np.ndarray:
    return np.array([flip(bundle[i], N) for i in range(len(bundle))])


def game_end(game: Board):
    """
    对残局分析气眼和联通分支的大小，判断是否 PASS
    """
    if game.referee.previous_pass:
        count = game.board.sum()
        v = (1 if count > 0 else -1 if count < 0 else 0) * game.color
        if v > 0:
            # 对手选择 pass，而我方优，选择 pass。
            return True
    n = game.n
    for i in range(n):
        for j in range(n):
            if game.board[i][j] == 0:
                for d0, d1 in [(-1,0), (1,0), (0,-1), (0,1)]:
                    if i + d0 < 0 or i + d0 >= n or j + d1 < 0 or j + d1 >= n:
                        continue
                    if game.board[i+d0, j+d1] == 0:
                        return False

    def liberty(x, y, c, visit, id0):
        if x < 0 or x >= n or y < 0 or y >= n:
            return 0
        if visit[x][y] == 1 or visit[x][y] == id0:
            return 0
        if game.board[x][y] == 0:
            visit[x][y] = id0
            return 1
        if game.board[x][y] != c:
            return 0
        visit[x][y] = 1
        return liberty(x-1, y, c, visit, id0) + liberty(x+1, y, c, visit, id0) \
            + liberty(x, y-1, c, visit, id0) + liberty(x, y+1, c, visit, id0)

    visit = np.zeros((n, n), dtype=np.int8)
    id0 = 1
    for i in range(n):
        for j in range(n):
            if not visit[i][j] and game.board[i][j] == game.color:
                id0 += 1
                c = game.board[i][j]
                lib = liberty(i, j, c, visit, id0)
                if lib >= 3:
                    return False
    visit.fill(0)
    for i in range(n):
        for j in range(n):
            if not visit[i][j] and game.board[i][j] == -game.color:
                id0 += 1
                c = game.board[i][j]
                lib = liberty(i, j, c, visit, id0)
                if lib <= 1:
                    return False
    return True


def evaluate(board: Board) -> tuple[int, int]:
    """
    一个人工设计的评估函数，用于评估当前局面的分数。这个评估函数包含了人类先验知识，
    但是不完全可靠，并没有覆盖所有可能的围棋情况，所以仅供参考。
    可以用于 self-play 初期 mcts 的 rollout ，来对策略网络和价值网络做预训练。
    """
    score = {1: 0, -1: 0}
    n = board.n
    visit = np.zeros((n, n), dtype=np.bool_)
    szX = np.zeros((n, n), dtype=np.int8)
    szO = np.zeros((n, n), dtype=np.int8)
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

    def calc_score(c: np.int8) -> int:
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
