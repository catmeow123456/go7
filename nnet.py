import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import HISTORY_SIZE, ACTION_SIZE, N

map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(map_location)

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

    def predict_multiple(self, input: np.ndarray):
        board = torch.from_numpy(input.astype(np.float32)).to(device)
        board = board.view(-1, CHANNELS_IN, N, N)

        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        pi, v = torch.exp(pi).cpu().numpy(), v.cpu().numpy()
        return [
            (pi[i], v[i])
            for i in range(len(pi))
        ]
