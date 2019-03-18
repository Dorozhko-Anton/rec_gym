import gym
from gym.spaces import Space
import numpy as np

class NDiscreteTuple(Space):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    self.observation_space = spaces.NDiscreteTuple(spaces.Discrete(2), 2)
    """
    def __init__(self, space, n):
        self.space = space
        self.n = n
        self.np_random = np.random.RandomState()
        super(NDiscreteTuple, self).__init__()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return tuple(self.np_random.choice(range(self.space.n), self.n))

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.n) and all(
            self.space.contains(part) for part in x) and (set(x) == self.n)

    def __repr__(self):
        return "NDiscreteTuple(" + ", ". join([str(self.space) for _ in range(self.n)]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [self.space.to_jsonable([sample[i] for sample in sample_n]) \
                for i in range(self.n)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[self.space.from_jsonable(sample_n[i]) for i in range(self.n)])]

    def __eq__(self, other):
        return (self.space == other.space) and (self.n == other.n)