class dotdict(dict):

    def __init__(self, x: dict):
        super().__init__(**x)
        self.__dict__ = self

    def __getattr__(self, name):
        return self[name]

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
