import random
import numpy as np
from typing import List
from collections import Counter
import time


class Entity:
    def __init__(self, location):
        self.location = location
        G[location - 1].append(self)
        self.data = None
        self.newest_received = None

    def __str__(self):
        return f"Entity({self.location}, {self.data})"

    def move(self, direction):
        if direction != 0:
            G[self.location - 1].remove(self)
            self.location += direction
            G[self.location - 1].append(self)

    def contact(self, new_data):
        if self.newest_received is None or new_data < self.newest_received:
            self.newest_received = new_data

    def update_data(self):
        if self.data is not None:
            self.data += 1
            if self.data >= t:
                self.data = None

        if self.newest_received is not None:
            if self.data is None or (self.newest_received < self.data and random.random() < r):
                self.data = self.newest_received + 1
                if self.data >= t:
                    self.data = None
            self.newest_received = None

    def random_walk(self):
        if self.location == 1:
            self.move(random.choice([0, 0, 1]))
        elif self.location == n:
            self.move(random.choice([-1, 0, 0]))
        else:
            self.move(random.choice([-1, 0, 1]))


def shift(arr):
    result = np.empty_like(arr)
    result[0] = 0
    result[1:] = arr[:-1]
    return result


def run():
    entities = []
    for _ in range(m):
        entities.append(Entity(random.randrange(n) + 1))
    times_seen = np.zeros(t, dtype=int)  # first element is useless, kept for simplicity of access
    aux_received_at_dcu = np.zeros(t, dtype=bool)

    repeats = np.zeros((t - 1) * m, dtype=int)
    # ^maximum amount of repeated deliveries is every entity always delivering every step of a data's lifetime
    first_hit = np.zeros(t, dtype=int)  # first element is useless, kept for simplicity of access
    ages = Counter()
    received_at_dcu = 0
    lost_after_receiving = 0
    for step in range(steps):
        if step == (steps // 100):
            # reset counters after 1% of the run, but keep the state of things
            repeats = np.zeros((t - 1) * m, dtype=int)
            first_hit = np.zeros(t, dtype=int)
            ages = Counter()
            received_at_dcu = 0
            lost_after_receiving = 0

        delta = step % tau
        # Update times_seen list and delivered repeats
        if (step - t) % tau == 0:
            repeats[times_seen[-1]] += 1
        if aux_received_at_dcu[-1]:
            received_at_dcu += 1
            if times_seen[-1] == 0:
                lost_after_receiving += 1
        times_seen = shift(times_seen)
        aux_received_at_dcu = shift(aux_received_at_dcu)

        for ent in entities:
            ages[ent.data] += 1  # needs to be done here because we change ent.data later (RSU delivery)

        # Establish contact with DCU
        for location in G[0:c + 1]:
            for ent in location:
                ent.contact(delta)

        # Establish contact with other entities
        for ent in entities:
            if ent.data is not None:
                if n - ent.location <= c:
                    if times_seen[ent.data] == 0:
                        first_hit[ent.data] += 1
                    times_seen[ent.data] += 1
                    ent.data = None
                else:
                    low = max(1, ent.location - c) - 1
                    high = min(n, ent.location + c) - 1
                    for location in G[low:high + 1]:
                        for other in location:
                            if other is not ent:
                                other.contact(ent.data)

        # Update position and data
        for ent in entities:
            ent.random_walk()
            ent.update_data()

        if not aux_received_at_dcu[delta]:
            for ent in entities:
                if ent.data == delta + 1:
                    aux_received_at_dcu[delta] = True
                    break

    return repeats, first_hit, ages, received_at_dcu, lost_after_receiving


if __name__ == '__main__':
    m = 2
    n = 6
    t = 10
    c = 1
    tau = 1

    r = 0.1
    if c >= n:
        raise SystemExit("c can't be >= n (otherwise DCU contacts RSU directly)")
    steps = 5000000

    random.seed(12345)

    G = [[] for _ in range(n)]  # type: List[List[Entity]]

    duration = time.time()
    repeats, first_hit, ages, received_at_dcu, lost_after_receiving = run()
    duration = time.time() - duration

    total_deliveries = np.multiply(repeats, np.arange(repeats.size)).sum()

    latency = np.multiply(first_hit, np.arange(t)).sum() / first_hit.sum()
    loss = repeats[0] / repeats.sum()
    loss_received = lost_after_receiving / received_at_dcu
    lost_at_DCU = 1 - received_at_dcu / repeats.sum()

    utilization = 1 - ages[None] / sum(ages.values())

    average_age = 0
    for age in ages:
        if age is not None:
            average_age += age * ages[age]
    average_age /= sum(ages.values()) - ages[None]

    one_delivery = repeats[1] / repeats.sum()
