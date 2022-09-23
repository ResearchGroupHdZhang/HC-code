import unittest

import numpy as np
from fact import FactSet
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector,RandomQuerySelector


class TestBaseQuerySelector(unittest.TestCase):
    def test_select(self):
        facts = FactSet(facts=np.array([[0, 0, 0, 0], [0, 0, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 1, 1],
                                        [0, 1, 0, 0], [0, 1, 0, 1],
                                        [0, 1, 1, 0], [0, 1, 1, 1],
                                        [1, 0, 0, 0], [1, 0, 0, 1],
                                        [1, 0, 1, 0], [1, 0, 1, 1],
                                        [1, 1, 0, 0], [1, 1, 0, 1],
                                        [1, 1, 1, 0], [1, 1, 1, 1]]),
                        prior_p=np.array([0.03, 0.06, 0.07, 0.04,
                                          0.09, 0.01, 0.11, 0.09,
                                          0.04, 0.04, 0.04, 0.05,
                                          0.06, 0.09, 0.07, 0.11]),
                        ground_true=2)
        accuracy = np.array([0.8, 0.8, 0.9, 0.85])  # 对应fact1, 3是0.8, 0.
        query_selector = BaseQuerySelector()
        selection_idxes, sub_facts, h = query_selector.select(facts, 2, accuracy)
        self.assertTrue(np.all(selection_idxes == np.array((0, 3))))


class TestGreedyQuerySelector(unittest.TestCase):   #改5
    def test_select(self):
        facts = FactSet(facts=np.array([[0, 0, 0, 0], [0, 0, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 1, 1],
                                        [0, 1, 0, 0], [0, 1, 0, 1],
                                        [0, 1, 1, 0], [0, 1, 1, 1],
                                        [1, 0, 0, 0], [1, 0, 0, 1],
                                        [1, 0, 1, 0], [1, 0, 1, 1],
                                        [1, 1, 0, 0], [1, 1, 0, 1],
                                        [1, 1, 1, 0], [1, 1, 1, 1]]),
                        prior_p=np.array([0.03, 0.06, 0.07, 0.04,
                                          0.09, 0.01, 0.11, 0.09,
                                          0.04, 0.04, 0.04, 0.05,
                                          0.06, 0.09, 0.07, 0.11]),
                        ground_true=2)
        accuracy = np.array([0.8, 0.8, 0.9, 0.85])  # 对应fact1, 3是0.8, 0.
        query_selector = GreedyQuerySelector()
        selection_idxes, sub_facts, h = query_selector.select(facts, 2, accuracy)
        print(selection_idxes)
        self.assertTrue(np.all(selection_idxes == np.array((0, 3))))


class TestRandomQuerySelector(unittest.TestCase):   #2.1
    def test_select(self):
        facts = FactSet(facts=np.array([[0, 0, 0, 0], [0, 0, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 1, 1],
                                        [0, 1, 0, 0], [0, 1, 0, 1],
                                        [0, 1, 1, 0], [0, 1, 1, 1],
                                        [1, 0, 0, 0], [1, 0, 0, 1],
                                        [1, 0, 1, 0], [1, 0, 1, 1],
                                        [1, 1, 0, 0], [1, 1, 0, 1],
                                        [1, 1, 1, 0], [1, 1, 1, 1]]),
                        prior_p=np.array([0.03, 0.06, 0.07, 0.04,
                                          0.09, 0.01, 0.11, 0.09,
                                          0.04, 0.04, 0.04, 0.05,
                                          0.06, 0.09, 0.07, 0.11]),
                        ground_true=2)
        accuracy = np.array([0.8, 0.8, 0.9, 0.85])  # 对应fact1, 3是0.8, 0.
        query_selector = RandomQuerySelector()
        selection_idxes, sub_facts, h = query_selector.select(facts, 2, accuracy)
        print("random_choice:",selection_idxes," 熵为:",h)