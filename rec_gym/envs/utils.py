import numpy as np


class User:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray):
        self.id = id
        self.embedding = embedding

    def __repr__(self):
        return "User (%s, %s)" % (str(self.id), str(self.embedding))


class Item:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray,
                 use_until: int):
        self.id = id
        self.embedding = embedding
        self.use_until = use_until

    def __repr__(self):
        return "Item (%s, %s)" % (str(self.id), str(self.embedding))
