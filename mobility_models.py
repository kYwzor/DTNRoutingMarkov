from scipy.sparse import coo_matrix, kron
from abc import ABC, abstractmethod


class BaseMobilityModel(ABC):
    # All mobility models should inherit from this class
    # Designed to avoid repeated pickling when using multiprocessing pools
    # see https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution
    name = "BaseMobilityModel"
    m = None
    n = None

    @classmethod
    @abstractmethod
    def setup(cls, model, settings):
        cls.m = model.m
        cls.n = model.n

    @classmethod
    @abstractmethod
    def movement(cls, L):
        pass


class RandomWalk(BaseMobilityModel):
    name = "RandomWalk"
    all_g = []

    @classmethod
    def setup(cls, model, settings):
        super().setup(model, settings)
        backwards = settings['a-']
        stay = settings['a0']
        forwards = settings['a+']
        assert backwards + stay + forwards == 1
        assert min(backwards, stay, forwards) > 0

        # There are only n different g, so we can cache all of them
        for b in range(cls.n):
            if b == 0:
                cls.all_g.append(coo_matrix(((backwards + stay, forwards), ((0, 0), (b, b + 1))), (1, cls.n)))
            elif b == cls.n - 1:
                cls.all_g.append(coo_matrix(((backwards, stay + forwards), ((0, 0), (b - 1, b))), (1, cls.n)))
            else:
                cls.all_g.append(coo_matrix(((backwards, stay, forwards), ((0, 0, 0), (b - 1, b, b + 1))), (1, cls.n)))

    @classmethod
    def movement(cls, L):
        G = cls.all_g[L[0] - 1]
        for i in range(1, cls.m):
            G = kron(G, cls.all_g[L[i] - 1], format="coo")
        return G


class ForwardWalk(BaseMobilityModel):
    name = "ForwardsWalk"
    all_g = None

    @classmethod
    def setup(cls, model, settings):
        super().setup(model, settings)
        forwards = settings['a+']
        assert 0 < forwards < 1
        stay = 1 - forwards

        # There are only n different g, so we can cache all of them
        cls.all_g = [coo_matrix(((stay, forwards), ((0, 0), (b, (b + 1) % cls.n))), (1, cls.n)) for b in range(cls.n)]

    @classmethod
    def movement(cls, L):
        G = cls.all_g[L[0] - 1]
        for i in range(1, cls.m):
            G = kron(G, cls.all_g[L[i] - 1], format="coo")
        return G


class CellularAutomaton(BaseMobilityModel):
    name = "CellularAutomaton"
    default = None
    speed = None

    @classmethod
    def setup(cls, model, settings):
        super().setup(model, settings)
        cls.speed = settings['speed']
        assert 0 < cls.speed

        # find index for positions [1, 2, 3...] to use as default when there are multiple entities in the same position
        index = 0
        for pos in range(1, cls.m):
            index += pos * cls.n ** (cls.m - 1 - pos)
        cls.default = coo_matrix(((1,), ((0,), (index,))), (1, cls.n ** cls.m))

    @classmethod
    def movement(cls, L):
        uniques = set()
        total = 0
        for i, b in enumerate(L):
            if b in uniques:
                return cls.default
            uniques.add(b)
            total += (L[(i + 1) % cls.m] - b) % cls.n
        if total != cls.n:
            return cls.default

        prob = 1 / ((L[1 % cls.m] - L[0]) % cls.n) ** cls.speed
        G = coo_matrix(((prob, 1 - prob), ((0, 0), (L[0] - 1, L[0] % cls.n))), (1, cls.n))
        for i in range(1, cls.m):
            prob = 1 / ((L[(i + 1) % cls.m] - L[i]) % cls.n) ** cls.speed
            G = kron(G, coo_matrix(((prob, 1 - prob), ((0, 0), (L[i] - 1, L[i] % cls.n))), (1, cls.n)), format="coo")
        return G
