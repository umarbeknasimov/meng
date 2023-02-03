from foundations.step import Step
from foundations import paths

class MetricLogger:
    def __init__(self):
        self.log = {}
    
    def add(self, name: str, step: Step, value: float):
        self.log[(name, step.iteration)] = value
    
    def get(self, name: str, step: Step):
        return self.log((name, step.iteration))

    def __str__(self):
        return '\n'.join('{},{},{}'.format(k[0], k[1], v) for k, v in self.log.items())
    
    @staticmethod
    def create_from_str(s: str):
        logger = MetricLogger()
        if len(s.strip()) == 0:
            return logger
        rows = [row.split(',') for row in s.split('\n')]
        logger.log = {(name, int(iteration)): value for name, iteration, value in rows}
        return logger     
    
    @staticmethod
    def create_from_file(filename: str):
        with open(paths.logger(filename)) as f:
            as_str = f.read()
        return MetricLogger.create_from_str(as_str)
    
    def save(self, location):
        with open(paths.logger(location), 'w') as f:
            f.write(str(self))
    
    def get_data(self, desired_name):
        data = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, data[k]) for k in sorted(data.keys())]
    



