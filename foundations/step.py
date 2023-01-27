class Step:
    """ represents a step in training
    can be either the total # of iterations or epoch & iteration within that epoch
    source: https://github.com/facebookresearch/open_lth/blob/main/foundations/step.py
    """

    def __init__(self, iteration : int, iterations_per_epoch: int):
        if iteration < 0: raise ValueError('iteration must be >=0')
        if iterations_per_epoch < 0: raise ValueError('iteration per epoch must be >= 0')
        self._iteration = iteration
        self._iterations_per_epoch = iterations_per_epoch
    
    @staticmethod
    def zero(iterations_per_epoch: int) -> 'Step':
        return Step(0, iterations_per_epoch)
    
    @staticmethod
    def from_iteration(iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(iteration, iterations_per_epoch)
    
    @staticmethod
    def from_epoch(epoch: int, iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(epoch * iterations_per_epoch + iteration, iterations_per_epoch)
    
    @staticmethod
    def from_str(s: str, iterations_per_epoch: int) -> 'Step':
        """create step from a string"""
        if 'ep' in s:
            ep = int(s.split('ep')[0])
            if s != '{}ep'.format(ep): raise ValueError('malformed string step: {}'.format(s))
            return Step.from_epoch(ep, 0, iterations_per_epoch)

    
    @property
    def iteration(self):
        """total number of iterations completed so far"""
        return self._iteration
    
    @property
    def it(self):
        """number of iterations completed within current epoch"""
        return self._iteration % self._iterations_per_epoch
    
    @property
    def ep(self):
        """current epoch in training"""
        return self._iteration // self._iterations_per_epoch
    
    def _check(self, other):
        if not isinstance(other. Step):
            raise ValueError('invalid type for other: {}'.format(type(other)))
        elif self._iterations_per_epoch != other._iterations_per_epoch:
            raise ValueError('cannot compare steps with different iterations per epoch')

    def __eq__(self, other):
        return self._iteration == other._iteration

    def __lt__(self, other):
        return self._iteration < other._iteration
    
    def __le__(self, other):
        return self._iteration <= other._iteration
    
    def __gt__(self, other):
        return self._iteration > other._iteration
    
    def __ge__(self, other):
        return self._iteration >= other._iteration
    
    def __str__(self):
        return 'iteration {}, iterations per epoch {}'.format(self._iteration, self._iterations_per_epoch)


