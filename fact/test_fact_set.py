import unittest

from fact import FactSet
from worker import Worker, BaseAnswerSampler
import numpy as np


class TestFactSet(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.fact_case1 = FactSet(facts=np.array([[0, 0, 0, 0], [0, 0, 0, 1],
                                                  [0, 0, 1, 0], [0, 0, 1, 1],
                                                  [0, 1, 0, 0], [0, 1, 0, 1],
                                                  [0, 1, 1, 0], [0, 1, 1, 1],
                                                  [1, 0, 0, 0], [1, 0, 0, 1],
                                                  [1, 0, 1, 0], [1, 0, 1, 1],
                                                  [1, 1, 0, 0], [1, 1, 0, 1],
                                                  [1, 1, 1, 0], [1, 1, 1, 1]]),
                                  prior_p=np.array([
                                      0.03, 0.06, 0.07, 0.04, 0.09, 0.01, 0.11,
                                      0.09, 0.04, 0.04, 0.04, 0.05, 0.06, 0.09,
                                      0.07, 0.11
                                  ]),
                                  ground_true=2)

    def test_compute_ans_p(self):
        facts = self.fact_case1
        accuracy = np.array([0.8, 0.7, 0.9, 0.85])  # 对应fact1, 3是0.8, 0.9
        ans = np.array([1, 0])
        p, _ = facts.compute_ans_p(ans, [0, 2], accuracy)
        self.assertAlmostEquals(p, 0.19,
                                msg=f"p={p} not almost equal to 0.2276")

    def test_compute_entropy(self):
        facts = self.fact_case1
        subfact1 = facts.get_subset([0, 1])  # entropy 1.993
        subfact2 = facts.get_subset([0, 2])  # entropy 1.982
        subfact3 = facts.get_subset([0, 3])  # entropy 1.997

        entropy1 = subfact1.compute_entropy()
        entropy1_ = subfact1.compute_ansset_entropy(np.array([0.8, 0.8]))

        entropy2 = subfact2.compute_entropy()
        entropy2_ = subfact2.compute_ansset_entropy(np.array([0.8, 0.8]))

        entropy3 = subfact3.compute_entropy()
        entropy3_ = subfact3.compute_ansset_entropy(np.array([0.8, 0.8]))

        self.assertAlmostEquals(entropy2, 1.992)
        self.assertAlmostEquals(entropy1, 1.993)
        self.assertAlmostEquals(entropy3, 1.997)


