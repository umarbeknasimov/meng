from foundations.step import Step
from foundations import paths

class IdsLogger:
    def __init__(self):
        self.log = {}
    
    def add(self, name: str, step: Step, ids: list):
        self.log[(name, step.iteration)] = ids.copy()
    
    def get(self, name: str, step: Step):
        return self.log[(name, step.iteration)]
    
    def has(self, name: str, step: Step):
        return (name, step.iteration) in self.log

    def __str__(self):
        return '\n'.join('{}.{}.{}'.format(k[0], k[1], ','.join([str(v_i) for v_i in v])) for k, v in self.log.items())
    
    @staticmethod
    def create_from_str(s: str):
        logger = IdsLogger()
        if len(s.strip()) == 0:
            return logger
        rows = [row.split('.') for row in s.split('\n')]
        logger.log = {(name, int(iteration)): [int(value) for value in values.split(',')] for name, iteration, values in rows}
        return logger     
    
    @staticmethod
    def create_from_file(filename: str):
        with open(paths.ids_logger(filename)) as f:
            as_str = f.read()
        return IdsLogger.create_from_str(as_str)
    
    def save(self, location):
        with open(paths.ids_logger(location), 'w') as f:
            f.write(str(self))
    
    def get_data(self, desired_name):
        data = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, data[k]) for k in sorted(data.keys())]