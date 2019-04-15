import numpy as np


def mrr(x):
    """
    mean reciprocal rank

    x : binary vector of relevance
    """
    x = np.asarray(x).nonzero()[0]
    return 1./(x[0] + 1) if len(x) > 0 else 0


def precision(x):
    """
    precision of binary relevance

    x : binary vector of relevance
    """
    return np.mean(x)


def precision_at_k(x, k):
    assert k >= 1
    assert len(x) >= k
    x = np.asarray(x)[:k] != 0
    return np.mean(x)


def average_precision(x):
    """
    precision for each possible K
    """
    x = np.asarray(x) != 0
    res = [precision_at_k(x, k+1) for k in range(len(x))]
    return np.mean(res)


def dcg_at_k(x, k):
    x = np.asfarray(x)[:k]
    if x.size:
        return np.sum(x / np.log2(np.arange(2, x.size + 2)))
    return 0


def ndcg_at_k(x, k):
    dcg_max = dcg_at_k(sorted(x, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(x, k) / dcg_max

# TODO: precision, recall, recall@K,
# TODO: Coverage, Personalization, IntraList similarity
