from functools import lru_cache
from typing import List, Tuple
import numpy as np
# import resource
from dataloader import IncreEntropyDataloader, NormalDataloader

@lru_cache(1)
def read_raw_data(num_section: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    with open("rte_final1.txt", "r") as f:
        raw_lines = f.readlines()
    idxes = []
    prior_p = []
    labels = []
    index = 0
    while index != len(raw_lines):
        idx = raw_lines[index]
        index += 1
        p = raw_lines[index].strip().split(" ")
        p = [float(i) for i in p]
        index += 1
        l = raw_lines[index].strip().split(" ")
        l = [int(i) for i in l]
        idxes.append(int(idx))
        prior_p.append(p)
        labels.append(l)
        index += 1
    idxes, prior_p, labels = np.asarray(idxes), np.asarray(prior_p), np.asarray(labels)
    if num_section > 1:  # TODO 划分不了的单独组成一个小的
        # num_per_split = [num_section] * (num // num_section)
        # num_per_split += [num - num_section * (num // num_section)]
        idxes = np.split(idxes, num_section)
        prior_p = np.split(prior_p, num_section)
        labels = np.split(labels, num_section)
        return idxes, prior_p, labels
    return [idxes], [prior_p], [labels]


if __name__ == '__main__':
    a, b, c = read_raw_data(722)
    n = NormalDataloader(a, b, c)
    i = IncreEntropyDataloader(a, b, c)
    for i, p, g in i:
        print(i, p, g)
    # for i, p, g in n:
    #     print(i, p, g)
"""
normal
[266] [0.8] [1]
[934] [0.3] [0]
[961] [0.6] [1]
[1814] [0.9] [1]
[134] [0.4] [0]
[103] [0.3] [0]...
"""
"""
incre
[1848] [0.5] [0]
[1271] [0.5] [1]
[1324] [0.5] [0]
[1201] [0.5] [0]
[1321] [0.5] [0]
[2135] [0.5] [0]...
"""