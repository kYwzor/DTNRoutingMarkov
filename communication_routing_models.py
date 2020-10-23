import numpy as np
from abc import ABC, abstractmethod
import itertools as it
from scipy.sparse import kron, coo_matrix


class BaseCommunicationRoutingModel(ABC):
    # All communication models should inherit from this class
    # Designed to avoid repeated pickling when using multiprocessing pools
    # see https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution
    name = "BaseCommunicationRoutingModel"
    m = None
    n = None
    t = None
    c = None
    tau = None

    @classmethod
    @abstractmethod
    def setup(cls, model, settings):
        cls.m = model.m
        cls.n = model.n
        cls.t = model.t
        cls.c = model.c
        cls.tau = model.tau

    @classmethod
    @abstractmethod
    def communication(cls, L, D, delta):
        pass

    @classmethod
    def chunk(cls, tup):
        L, mob = tup
        gen = it.product(it.product(range(1, cls.t + 1), repeat=cls.m), range(cls.tau))
        com_size = cls.tau * cls.t ** cls.m
        data, col = zip(*(cls.communication(L, D, delta) for D, delta in gen))

        row = np.concatenate([np.full_like(col[i], i) for i in range(len(data))])
        data = np.concatenate(data)
        col = np.concatenate(col)

        com = coo_matrix((data, (row, col)), shape=(com_size, com_size))
        return kron(mob, com, format='coo')


class EpidemicRouting(BaseCommunicationRoutingModel):
    name = "EpidemicRouting"
    dtype = None
    probs = None
    data_start = None
    col_start = None
    col_cache = None

    @classmethod
    def setup(cls, model, settings):
        super().setup(model, settings)
        r = settings['r']
        assert 0 <= r <= 1

        cls.dtype = np.int64 if cls.tau * cls.t ** cls.m > np.iinfo('int32').max else np.int32
        cls.probs = np.array((1 - r, r))
        cls.data_start = np.ones(1)
        cls.col_start = np.zeros(1, dtype=cls.dtype)

        cls.col_cache = {}
        for i in range(1, cls.t - 1):
            for j in range(i):
                cls.col_cache[(i, j)] = np.array((i, j))

    @classmethod
    def communication(cls, L, D, delta):
        data = cls.data_start
        col = cls.col_start
        for i in range(cls.m):
            if cls.n - L[i] <= cls.c:
                Ki = cls.t
            else:
                Ki = min(D[i] + 1, cls.t)

            if L[i] - 1 <= cls.c:
                Ri = delta + 1
            else:
                Ri = cls.t
                for j in range(cls.m):
                    if i != j and abs(L[i] - L[j]) <= cls.c < cls.n - L[j] and D[j] + 1 < Ri:
                        Ri = D[j] + 1

            if Ri < Ki < cls.t:
                data = (data.repeat(2).reshape(-1, 2) * cls.probs).reshape(-1)
                col = ((col * cls.t).repeat(2).reshape(-1, 2) + cls.col_cache[(Ki - 1, Ri - 1)]).reshape(-1)
            else:
                col = col * cls.t + (min(Ki, Ri) - 1)
        col = col * cls.tau + (delta + 1) % cls.tau
        return data, col
