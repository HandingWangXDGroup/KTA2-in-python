class Individual:
    def __init__(self, x, problem):
        self.x = x
        self.problem = problem
        self.obj = self.problem.evaluate(x)
