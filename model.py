import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Tuple
from game import Board, ACTION_SIZE, PASS, HISTORY_SIZE, N, evaluate
map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(map_location)
EPS = 1e-8
CHANNELS_IN = HISTORY_SIZE * 2 + 1

class NNet(nn.Module):
    """
    策略网络：
    输入棋盘当前状态 
    CHANNELS_IN*N*N 的数组，第一层是我方颜色位置，第二层是对方颜色位置，.... （保存了 HISTORY_SIZE 次历史记录）第零层是合法下棋位置
    输出动作概率分布，以及状态胜率。
    """
    def __init__(self, dropout, num_channels, hidden_size):
        super().__init__()

        self.conv1 = nn.Conv2d(CHANNELS_IN, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels * 9, hidden_size)
        self.fc_bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_bn2 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, ACTION_SIZE)

        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, s):
        s = s.view(-1, CHANNELS_IN, N, N)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.fc1.in_features)

        s = self.dropout(F.relu(self.fc_bn1(self.fc1(s))))
        s = self.dropout(F.relu(self.fc_bn2(self.fc2(s))))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, input: np.ndarray):
        board = torch.from_numpy(input.astype(np.float32)).to(device)
        board = board.view(CHANNELS_IN, N, N)

        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]


class MCTS:
    nnet: NNet
    cpuct: float = 1.0

    # 记忆化 End state 和 valid moves
    Ps: Dict[bytes, np.ndarray] = {}    # 策略网络输出的概率分布

    Ns: Dict[bytes, int] = {}       # 状态 s 被访问的次数
    Qsa: Dict[Tuple[bytes, int], float] = {}  # stores Q values for s,a (as defined in the paper)
    Nsa: Dict[Tuple[bytes, int], int] = {}      # (s, a) 被访问的次数

    def __init__(self, nnet: NNet = None):
        self.nnet = nnet

    def best_move(self, game: Board, timeout: float) -> int:
        if timeout > 0:
            start_time = time.perf_counter()
            count = 0
            while time.perf_counter() - start_time < timeout:
                count += 1
                self._search(game.copy(), np.count_nonzero(game.board))
            print('count=', count)

        s = game.hashed_state()
        vs = game.legal_moves_input()
        # 选择探索次数最多的动作
        best_action = -1
        best_value = -1
        for a in range(ACTION_SIZE):
            if vs[a] > 0:
                nsa = self.Nsa.get((s, a), 0)
                if nsa > best_value:
                    best_value = nsa
                    best_action = a
        # print(best_action, best_value)
        return best_action

    def query_v(self, game: Board, action) -> float:
        s = game.hashed_state()
        return self.Qsa.get((s, action), None)

    def _search(self, game: Board, depth: int = 0) -> float:
        if hasattr(game, 'winner'):
            v = float(game.winner * game.color)
            return -v
        # if depth >= 45:
        #     res = evaluate(game)
        #     s = 1 if res[1] > res[-1] else (-1 if res[1] < res[-1] else 0)
        #     v = s * game.color
        #     return -v

        s = game.hashed_state()
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
            
            if s == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
                ps[game.move2int(3, 3)] *= 1000
            if s == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
                ps[game.move2int(2, 3)] *= 1000
                ps[game.move2int(3, 2)] *= 1000
                ps[game.move2int(2, 4)] *= 1000
                ps[game.move2int(4, 2)] *= 1000
            self.Ps[s] = ps
            self.Ns[s] = 0

            # if random.random() < 0.2:
            #     v1, v2 = evaluate(game)
            #     v = 1 if v1 > v2 else (-1 if v1 < v2 else 0)
            #     v = v * game.color
            #     return -float(v)
            return -float(v)

        # 选择动作
        ps = self.Ps[s]
        ns = self.Ns[s]
        cpuct = self.cpuct

        best_uct = -float("inf")
        best_act = -1

        

        # pick the action with the highest upper confidence bound
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
        v = self._search(game, depth + 1)

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
