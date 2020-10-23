import numpy as np
import pickle


class TPMSimulator:
    def __init__(self, m, n, t, c, tau, P):
        self.m = m
        self.n = n
        self.t = t
        self.c = c
        self.tau = tau
        self.sequence = None
        self.size = P.shape[0]
        self.tpm = P

    def simulate(self, steps):
        self.sequence = np.empty(steps, dtype=int)
        np.random.seed(0)

        self.sequence[0] = np.random.choice(self.size)
        for step in range(1, steps):
            if not step % 10000:
                print(step)
            roll = np.random.random_sample()

            # Could improve performance by caching sparse cumulative sum and using bisect / numpy.searchsorted
            # or by caching self.tpm.nonzero() and masking for needed row, for use with np.choice()
            # but this will work
            row = self.tpm[self.sequence[step - 1]]
            prob = 0
            for index, data in zip(row.indices, row.data):
                prob += data
                if roll <= prob:
                    break
            self.sequence[step] = index

    def calculate_simulated_pi(self):
        unique, counts = np.unique(self.sequence, return_counts=True)
        simulated_pi = np.zeros(self.size)
        simulated_pi[unique] = counts / counts.sum()
        return simulated_pi

    def index_to_state(self, index):
        assert 0 <= index < self.size

        delta = index % self.tau
        index //= self.tau

        data = np.empty((self.m,), dtype=int)
        for i in reversed(range(self.m)):
            data[i] = (index % self.t) + 1
            index //= self.t

        positions = np.empty((self.m,), dtype=int)
        for i in reversed(range(self.m)):
            positions[i] = (index % self.n) + 1
            index //= self.n
        return positions, data, delta

    def simulation_loss(self):
        total = 0
        lost = 0
        for pos in range(self.sequence.size - (self.t - 1)):
            _, _, delta = self.index_to_state(self.sequence[pos])
            if delta != 0:
                continue
            total += 1
            found = False
            for a in range(self.t - 1):
                L, D, _ = self.index_to_state(self.sequence[pos + a])
                for i in range(self.m):
                    if self.n - L[i] <= self.c and D[i] == a:
                        found = True
                        break
                if found:
                    break
            if not found:
                lost += 1
        return lost / total

    def simulation_age_distribution(self):
        ages = np.zeros(self.t)
        for state in self.sequence:
            _, D, _ = self.index_to_state(state)
            for age in D:
                ages[age - 1] += 1
        return ages / ages.sum()


def compare_pi(calculated_pi, simulated_pi):
    diff = np.absolute(calculated_pi - simulated_pi)
    return diff.mean(), diff.max()


if __name__ == '__main__':
    file = "pickled/tpm/m2_n11_t20_c2_tau10_mmForwardsWalk_a+0.6666666666666666_cmEpidemicRouting_r0.76.pkl"
    with open(file, 'rb') as file:
        loaded_tpm = pickle.load(file)
    simulation = TPMSimulator(2, 11, 20, 2, 10, loaded_tpm)
    simulation.simulate(1000000)
    print("Simulation ended")
    simulated_pi = simulation.calculate_simulated_pi()
    print("Simulated pi")
    sim_loss = simulation.simulation_loss()
    print("Simulated loss")
    sim_age = simulation.simulation_age_distribution()
    print("Simulated age")
    sim_avg_age = np.multiply(sim_age[:-1], np.arange(1, simulation.t)).sum() / (1 - sim_age[-1])
    print("Simulated average age")
