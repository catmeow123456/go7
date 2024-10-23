import time
import math
import numpy as np
from typing import Dict, Tuple
from game import Board, ACTION_SIZE, game_end, PASS
from nnet import NNet
from utils import Args

EPS = 1e-8


class MCTS:
    nnet: NNet
    cpuct: float = 1.0
    args: Args

    Ps: Dict[bytes, np.ndarray]    # 记忆化策略网络输出的概率分布
    Ns: Dict[bytes, int]       # 状态 s 被访问的次数
    Qsa: Dict[Tuple[bytes, int], float]  # stores Q values for s,a (as defined in the paper)
    Nsa: Dict[Tuple[bytes, int], int]      # (s, a) 被访问的次数

    def __init__(self, nnet: NNet = None, args: Args = None):
        self.nnet = nnet
        self.args = args
        self.pipe = None
        self.valids = {}

        self.Ps = {}
        self.Ns = {}
        self.Qsa = {}
        self.Nsa = {}

    def query_v(self, game: Board, action) -> float:
        s = game.hashed_state
        return self.Qsa.get((s, action), None)

    def getActionProb(self, game: Board, temp=1) -> np.ndarray:
        """
        通过 MCTS 搜索得到动作概率分布
        :param game: 当前棋盘状态
        :param temp: 温度参数
        :return: 动作概率分布
        """
        if game_end(game):
            arr = np.zeros(ACTION_SIZE)
            arr[PASS] = 1
            return arr
        s = game.hashed_state
        valids = self.valids.get(s, None)
        if valids is None:
            valids = game.legal_moves_input()
        if valids.sum() == 1:
            # 无路可走，只能 PASS
            arr = np.zeros(ACTION_SIZE)
            arr[PASS] = 1
            return arr
        for _ in range(self.args.numMCTSSims):
            self._search(game.copy())
        Ns = np.array([self.Nsa.get((s, a), 0) for a in range(ACTION_SIZE)])
        if temp == 0:
            best_as = np.argwhere(Ns == np.max(Ns)).flatten()
            best_a = np.random.choice(best_as)
            probs = np.zeros(ACTION_SIZE)
            probs[best_a] = 1
        else:
            Ns = Ns ** (1 / temp)
            Ns_sum = np.sum(Ns)
            probs = Ns / Ns_sum
        return probs

    def _search(self, game: Board, depth=0) -> float:
        if depth>200:
            print(f"depth = {depth}\n{game}")
        if hasattr(game, 'winner'):
            v = float(game.winner * game.color)
            return -v
        if game_end(game):
            game.place(-1, -1)
            # 已经到终局，选择 pass
            return -self._search(game, depth+1)
        # 其他情况不选择 pass
        s = game.hashed_state
        valids = self.valids.get(s, None)
        if valids is None:
            valids = game.legal_moves_input()
        if valids.sum() == 1:
            game.place(-1, -1)
            return -self._search(game, depth+1)
        # 叶子结点 （ 如果没有访问过，更新 Ps, Ns ）扩展
        if s not in self.Ps:
            if self.nnet is not None:
                if self.pipe is not None:
                    self.pipe.send(game.bundled_input(valids))
                    ps, v = self.pipe.recv()
                else:
                    ps, v = self.nnet.predict(game.bundled_input(valids))
                ps = ps * valids
                ps_sum = np.sum(ps)
                if ps_sum > 0:
                    ps /= ps_sum  # renormalize
                else:  # All valid moves were masked.
                    ps = ps + valids
                    ps /= np.sum(ps)
            else:
                ps = valids / np.sum(valids)
            self.Ps[s] = ps
            self.Ns[s] = 0
            return -float(v)

        # 选择动作
        ps = self.Ps[s]
        ns = self.Ns[s]
        cpuct = self.cpuct

        best_uct = -float("inf")
        best_act = -1

        # 选择 UCB 最大的动作
        for a in range(ACTION_SIZE):
            if a == PASS or not valids[a]:
                continue
            qsa = self.Qsa.get((s, a), None)
            if qsa is not None:
                u = qsa + cpuct * ps[a] * math.sqrt(ns) / (1 + self.Nsa[s, a])
            else:
                u = cpuct * ps[a] * math.sqrt(ns + EPS)  # Q = 0
            if u > best_uct:
                best_uct = u
                best_act = a
        assert best_act >= 0
        a = best_act

        game.place(*game.int2move(a))
        v = self._search(game, depth+1)

        qsa = self.Qsa.get((s, a), None)
        if qsa is not None:
            nsa = self.Nsa[s, a]
            self.Qsa[s, a] = (nsa * qsa + v) / (1 + nsa)
            self.Nsa[s, a] = nsa + 1
        else:
            self.Qsa[s, a] = v
            self.Nsa[s, a] = 1

        self.Ns[s] += 1
        return -v
