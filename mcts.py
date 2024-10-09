import time
import math
import numpy as np
from typing import Dict, Tuple
from game import Board, ACTION_SIZE, evaluate
from nnet import NNet
from utils import dotdict

EPS = 1e-8


class MCTS:
    nnet: NNet
    cpuct: float = 1.0
    args: dotdict

    Ps: Dict[bytes, np.ndarray] = {}    # 记忆化策略网络输出的概率分布
    Ns: Dict[bytes, int] = {}       # 状态 s 被访问的次数
    Qsa: Dict[Tuple[bytes, int], float] = {}  # stores Q values for s,a (as defined in the paper)
    Nsa: Dict[Tuple[bytes, int], int] = {}      # (s, a) 被访问的次数

    def __init__(self, nnet: NNet = None, args: dotdict = None):
        self.nnet = nnet
        self.args = args

    def best_move(self, game: Board, timeout: float) -> int:
        if timeout > 0:
            start_time = time.perf_counter()
            count = 0
            while time.perf_counter() - start_time < timeout:
                count += 1
                self._search(game.copy(), np.count_nonzero(game.board))
            print('count=', count)

        s = game.hashed_state
        # 选择探索次数最多的动作
        best_action = -1
        best_value = -1
        for a in range(ACTION_SIZE):
            nsa = self.Nsa.get((s, a), None)
            if nsa is not None and nsa > best_value:
                best_value = nsa
                best_action = a
        # print(best_action, best_value)
        return best_action

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
        for _ in range(self.args.numMCTSSims):
            self._search(game.copy())
        s = game.hashed_state
        Ns = np.array([self.Nsa.get((s, a), 0) for a in range(ACTION_SIZE)])
        if temp == 0:
            best_as = np.argwhere(Ns == np.max(Ns)).flatten()
            best_a = np.random.choice(best_as)
            probs = np.zeros(ACTION_SIZE)
            probs[best_a] = 1
            return probs
        Ns = Ns ** (1 / temp)
        Ns_sum = np.sum(Ns)
        return Ns / Ns_sum

    def _search(self, game: Board) -> float:
        if hasattr(game, 'winner'):
            v = float(game.winner * game.color)
            return -v
        s = game.hashed_state
        valids = game.legal_moves_input()

        # 叶子结点 （ 如果没有访问过，更新 Ps, Ns ）扩展
        if s not in self.Ps:
            if self.nnet is not None:
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
            # 一些人类先验：不到万不得已不要加
            # if s == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
            #     ps[game.move2int(3, 3)] *= 1000
            # if s == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
            #     ps[game.move2int(2, 3)] *= 1000
            #     ps[game.move2int(3, 2)] *= 1000
            #     ps[game.move2int(2, 4)] *= 1000
            #     ps[game.move2int(4, 2)] *= 1000
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
            if not valids[a]:
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
        v = self._search(game)

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
