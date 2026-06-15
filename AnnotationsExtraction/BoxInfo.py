class BoxInfo:
    def __init__(self, line):
        self.parts = line.strip().split()
        self.playerId = self.parts.pop(0)
        self.category = self.parts.pop()
        self.xMin, self.yMin, self.xMax, self.yMax = map(int, self.parts[0:4])
        self.frameId = self.parts[4]
        self.lost, self.grouping, self.generated = map(int, [self.parts[5], self.parts[6], self.parts[7]])
