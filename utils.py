class Args:
    numIters: 1000  # 策略迭代 numIters 次数
    numEps: 100  # 进行完整的 numEps 轮游戏
    numMCTSSims: 1000  # 在每一轮游戏的每一个节点，运行 numMCTSSims 次 MCTS 模拟后再采样行动和获取样本
    maxlenOfQueue: 100000  # maxlenOfQueue 个样本为一组
    numItersForTrainExamplesHistory: 10  # 保留最近 numItersForTrainExamplesHistory 组样本，将他们混合后 shuffle 出训练集
    arenaCompare: 50  # 与历史模型对弈 arenaCompare 次，用于评估新模型的优劣
    updateThreshold: 0.55  # 新模型胜率超过 updateThreshold 时，接受新模型
    def __init__(self) -> None:
        self.numIters = 1000
        self.numEps = 100
        self.numMCTSSims = 500
        self.maxlenOfQueue = 100000
        self.numItersForTrainExamplesHistory = 10
        self.arenaCompare = 50
        self.updateThreshold = 0.55

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

