import pickle
import numpy as np
import itertools as it
import os
import time
import matplotlib.pyplot as plt
import ctypes
from scipy.sparse import vstack, identity, csr_matrix
from scipy.sparse.linalg import lgmres
from scipy.sparse.csgraph import connected_components
from typing import Type
from mobility_models import BaseMobilityModel, RandomWalk, ForwardWalk, CellularAutomaton
from communication_routing_models import BaseCommunicationRoutingModel, EpidemicRouting
from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray

FORCE_REBUILD = False


class MarkovChain:
    m: int = None  # amount of entities
    n: int = None  # amount of locations
    t: int = None  # time-to-live (ttl)
    c: int = None  # contact range
    tau: int = None  # sampling period
    mm: Type[BaseMobilityModel] = None  # mobility model
    ms: dict = None  # mobility parameters
    cm: Type[BaseCommunicationRoutingModel] = None  # communication model
    cs: dict = None  # communication settings

    size: int = None  # amount of states (s)
    tpm: csr_matrix = None  # transition probability matrix \mathbf{P}
    pi: np.ndarray = None  # stationary state probability vector

    delivery_counter: np.ndarray = None
    delivery_states: np.ndarray = None  # \mathcal{D} derived from delivery_counter
    single_delivery_states: np.ndarray = None  # \mathcal{O} derived from delivery_counter
    state_age_counter: np.ndarray = None  # \mathcal{C} can be deduced from this

    mob_filename: str = None
    tpm_filename: str = None
    steady_filename: str = None

    @classmethod
    def update(cls, m, n, t, c, tau, mm, ms, cm, cs):
        assert m >= 1
        assert n >= 2
        assert t >= 2
        assert 0 <= c < n - 1  # Distance from DCU to RSU has to be bigger than c otherwise direct contact is possible
        assert 1 <= tau < t

        cls.size = (n * t) ** m * tau
        assert cls.size <= np.iinfo('int64').max + 1  # coo_matrix row/column indices need to fit in np.int64

        cls.m = m
        cls.n = n
        cls.t = t
        cls.c = c
        cls.tau = tau
        cls.mm = mm
        cls.ms = ms
        cls.cm = cm
        cls.cs = cs

        cls.mob_filename = f"pickled/mobility/m{cls.m}_n{cls.n}" \
                           f"_mm{cls.mm.name}_{'_'.join(f'{key}{value}' for key, value in cls.ms.items())}.pkl"

        detailed_name = f"m{cls.m}_n{cls.n}_t{cls.t}_c{cls.c}_tau{cls.tau}" \
                        f"_mm{cls.mm.name}_{'_'.join(f'{key}{value}' for key, value in cls.ms.items())}" \
                        f"_cm{cls.cm.name}_{'_'.join(f'{key}{value}' for key, value in cls.cs.items())}.pkl"
        cls.tpm_filename = f"pickled/tpm/{detailed_name}"
        cls.steady_filename = f"pickled/steady/{detailed_name}"

    @classmethod
    def index_to_state(cls, index):
        assert 0 <= index < cls.size

        delta = index % cls.tau
        index //= cls.tau

        data = np.empty((cls.m,), dtype=int)
        for i in reversed(range(cls.m)):
            data[i] = (index % cls.t) + 1
            index //= cls.t

        positions = np.empty((cls.m,), dtype=int)
        for i in reversed(range(cls.m)):
            positions[i] = (index % cls.n) + 1
            index //= cls.n
        return positions, data, delta

    @classmethod
    def build_tpm(cls):
        loaded_mob = False
        if not FORCE_REBUILD:
            try:
                with open(cls.tpm_filename, 'rb') as file:
                    cls.tpm = pickle.load(file)
                    if __debug__:
                        print("Loaded TPM from", cls.tpm_filename)
                    return
            except FileNotFoundError:
                pass
            try:
                with open(cls.mob_filename, 'rb') as file:
                    mob = pickle.load(file)
                    loaded_mob = True
                    if __debug__:
                        print("Loaded mobility from", cls.mob_filename)
            except FileNotFoundError:
                pass

        if __debug__:
            print(f"Calculating TPM of size {cls.size}")

        if not loaded_mob:
            cls.mm.setup(cls, cls.ms)
        cls.cm.setup(cls, cls.cs)

        with Pool() as pool:
            if not loaded_mob:
                mob = pool.map(cls.mm.movement, it.product(range(1, cls.n + 1), repeat=cls.m))
                os.makedirs(os.path.dirname(cls.mob_filename), exist_ok=True)
                with open(cls.mob_filename, 'wb') as file:
                    pickle.dump(mob, file, pickle.HIGHEST_PROTOCOL)
            chunks = pool.map(cls.cm.chunk,
                              ((j, mob[i]) for i, j in enumerate(it.product(range(1, cls.n + 1), repeat=cls.m))))

        P = vstack(chunks)
        assert P.shape == (cls.size, cls.size)

        P.eliminate_zeros()
        P = P.tocsr()
        assert connected_components(P, directed=True, connection='weak', return_labels=False) == 1
        cls.tpm = P
        if __debug__:
            print("TPM calculated")

        os.makedirs(os.path.dirname(cls.tpm_filename), exist_ok=True)
        with open(cls.tpm_filename, 'wb') as file:
            pickle.dump(cls.tpm, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def calculate_pi(cls):
        if not FORCE_REBUILD:
            try:
                with open(cls.steady_filename, 'rb') as file:
                    cls.pi = pickle.load(file)
                    if __debug__:
                        print("Loaded steady state from", cls.steady_filename)
                    return
            except FileNotFoundError:
                pass
        if __debug__:
            print("Calculating steady state")
        # Ax = b
        A = cls.tpm.transpose() - identity(cls.size)
        A = vstack([A[:-1, :], np.ones(cls.size)], format='csr')

        b = np.zeros(cls.size)
        b[-1] = 1

        x, info = lgmres(A, b, tol=1e-14, atol=1e-14, maxiter=5000)

        # Check if there are any negative values, but tolerate errors smaller than 1e-10
        assert np.allclose(x[x < 0], 0, rtol=0, atol=1e-10), "Steady State with negative values"
        assert info == 0, "The iterative method did not converge"
        cls.pi = x
        if __debug__:
            print("Steady state calculated")

        os.makedirs(os.path.dirname(cls.steady_filename), exist_ok=True)
        with open(cls.steady_filename, 'wb') as file:
            pickle.dump(cls.pi, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def categorize_state(cls, tup):
        x, (L, D, delta) = tup
        for i in range(cls.m):
            cls.state_age_counter[D[i] - 1][x] += 1
            if cls.n - L[i] <= cls.c and D[i] != cls.t:
                cls.delivery_counter[D[i] - 1][x] += 1

    @classmethod
    def create_state_sets(cls):
        # np.uint8 is used for counters to save memory, since if m>255 size assertions would fail
        shared_array = RawArray(ctypes.c_uint8, (cls.t - 1) * cls.size)
        shared_array = np.frombuffer(shared_array, dtype=np.uint8)
        cls.delivery_counter = shared_array.reshape((cls.t - 1, cls.size))  # One set for each age except expired

        shared_array = RawArray(ctypes.c_uint8, cls.t * cls.size)
        shared_array = np.frombuffer(shared_array, dtype=np.uint8)
        cls.state_age_counter = shared_array.reshape((cls.t, cls.size))

        states = enumerate(it.product(it.product(range(1, cls.n + 1), repeat=cls.m),
                                      it.product(range(1, cls.t + 1), repeat=cls.m),
                                      range(cls.tau)))
        with Pool() as pool:
            pool.map(cls.categorize_state, states)

        cls.delivery_states = cls.delivery_counter > 0
        cls.single_delivery_states = cls.delivery_counter == 1

    @classmethod
    def calculate_loss(cls):
        Z = cls.pi.copy()
        for a in range(1, cls.t):
            Z[cls.delivery_states[a - 1]] = 0
            Z = Z * cls.tpm
        return Z.sum() * cls.tau - cls.tau + 1

    @classmethod
    def calculate_lost_at_dcu(cls):
        C = cls.state_age_counter > 0
        Z = cls.pi.copy()
        for a in range(1, cls.tau + 1):
            Z[C[a - 1]] = 0
            Z = Z * cls.tpm
        return Z.sum() * cls.tau - cls.tau + 1

    @classmethod
    def calculate_latency(cls):
        Z = cls.pi.copy()
        Ja = np.empty(cls.t - 1)
        Ja[0] = Z[cls.delivery_states[0]].sum()

        for a in range(1, cls.t - 1):
            Z[cls.delivery_states[a - 1]] = 0
            Z = Z * cls.tpm

            Ja[a] = Z[cls.delivery_states[a]].sum()
        return np.multiply(Ja, np.arange(1, cls.t)).sum() / Ja.sum()

    @classmethod
    def calculate_one_copy(cls):
        total_prob = 0
        for a in range(1, cls.t):
            Z = cls.pi.copy()
            for q in range(1, cls.t):
                if a == q:
                    Z[~cls.single_delivery_states[q - 1]] = 0
                else:
                    Z[cls.delivery_states[q - 1]] = 0
                Z = Z * cls.tpm
            total_prob += Z.sum()
        return total_prob * cls.tau

    @classmethod
    def calculate_age_stats(cls):
        M = cls.pi * cls.state_age_counter
        age_distribution = M.sum(axis=1) / cls.m  # equivalent to w

        # multiply each age proportion (except t) by its age, sum then divide by their total proportion
        average_age = np.multiply(age_distribution[:-1], np.arange(1, cls.t)).sum() / (1 - age_distribution[-1])
        utilization = 1 - age_distribution[cls.t - 1]

        return age_distribution, average_age, utilization

    @classmethod
    def debug_state(cls, index, extensive=False):
        print(f"_____________\nDebugging state {index}:\n{cls.index_to_state(index)}")
        row = cls.tpm.getrow(index)
        print(f"\nCan transition to {row.indices.size} different states")
        for pos in range(row.indices.size):
            print("State", row.indices[pos], cls.index_to_state(row.indices[pos]), row.data[pos])

        col = cls.tpm.getcol(index)
        print(f"\nCan be reached from {col.indices.size} different states")
        if extensive:
            for pos in range(col.indices.size):
                print("State", col.indices[pos], cls.index_to_state(col.indices[pos]), col.data[pos])

    @classmethod
    def gather_metrics(cls):
        start_time = time.time()
        cls.build_tpm()
        tpm_runtime = time.time() - start_time
        if __debug__:
            print("TPM runtime:", tpm_runtime)

        start_time = time.time()
        cls.calculate_pi()
        pi_runtime = time.time() - start_time
        if __debug__:
            print("Steady state runtime:", pi_runtime)

        start_time = time.time()
        cls.create_state_sets()
        loss = cls.calculate_loss()
        lost_at_DCU = cls.calculate_lost_at_dcu()
        loss_received = (loss - lost_at_DCU) / (1 - lost_at_DCU)
        latency = cls.calculate_latency()
        one = cls.calculate_one_copy()
        distribution, average_age, utilization = cls.calculate_age_stats()

        metrics_runtime = time.time() - start_time
        if __debug__:
            print("Metrics runtime:", metrics_runtime)

        return (loss, lost_at_DCU, loss_received, latency, one, average_age, utilization, distribution), \
               (tpm_runtime, pi_runtime, metrics_runtime)


def visualize_sparse_matrix(m):
    # adapted from https://stackoverflow.com/a/22965622
    m = m.asformat('coo')
    sparsity = 1 - (m.nnz / np.prod(m.shape))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, ',', color='blue')
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_aspect('equal')
    plt.title(f"sparsity: {sparsity}")
    plt.savefig("sparsity")
    ax.figure.show()


if __name__ == '__main__':
    # Usage Example
    m = 2
    n = 6
    t = 10
    c = 2
    tau = 2

    # mm = RandomWalk
    # ms = {"a-": 1/3, "a0": 1/3, "a+": 1/3}
    mm = ForwardWalk
    ms = {"a+": 1 / 2}
    cm = EpidemicRouting

    # cs = {"r": 0.5}
    # MarkovChain.update(m, n, t, c, tau, mm, ms, cm, cs)
    # visualize_sparse_matrix(MarkovChain.tpm)

    best_loss = 1
    best_r = -1
    results = []
    for r in np.linspace(0, 1, 101):
        cs = {"r": r}
        MarkovChain.update(m, n, t, c, tau, mm, ms, cm, cs)
        metrics, runtimes = MarkovChain.gather_metrics()
        print(metrics)
        results.append((r, metrics))
        if best_loss > metrics[0]:
            best_loss = metrics[0]
            best_r = r

    print(f"best_r = {best_r}")

    with open("results.pkl", 'wb') as file:
        pickle.dump((f"m={m}_n={n}_t={t}_c={c}_tau={tau}_a+={ms['a+']:.3f}", results), file, pickle.HIGHEST_PROTOCOL)
