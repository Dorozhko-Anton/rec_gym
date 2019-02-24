import numpy as np

class User:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray):
        self.id = id
        self.embedding = embedding


class Item:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray,
                 use_until: int):
        self.id = id
        self.embedding = embedding
        self.use_until = use_until
