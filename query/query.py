import imp
import numpy as np
from itertools import combinations

from regex import subf
from fact import FactSet
from typing import Tuple
import abc
import copy


class QuerySelector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(self, facts: FactSet,
               num: int,
               worker_accuracy: np.array) -> Tuple[np.ndarray, "FactSet", float]:
        """
        根据该类的策略，从facts中选取num个fact
        :param facts: 需要被选择作为问题的事实集
        :param num: 选取fact作为问题的数量
        :param worker_accuracy: 工人的回答准确率
        :return: 返回facts的一个子集, 包括其相对于原来的索引
        """
        raise NotImplemented("implement me")


class BaseQuerySelector(QuerySelector):
    """
    暴力法的问题选择器
    """

    def select(self, facts: FactSet,
               num: int,
               worker_accuracy: np.ndarray
               ) -> Tuple[np.ndarray, "FactSet", float]:
        num_fact: int = facts.num_fact()
        assert num <= num_fact
        selections = combinations(range(num_fact), num)
        max_h: float = -float('inf')
        max_selection: Tuple[int] = (0,) * num_fact
        for selection in selections:
            sub_facts = facts.get_subset(list(selection))
            h = 0.
            # 计算answer set的墒
            for i in range(len(sub_facts)):
                # print("sub_facts",sub_facts[i])
                p_ans, _ = sub_facts.compute_ans_p(sub_facts[i],
                                                list(range(sub_facts.num_fact())),
                                                worker_accuracy[:,list(selection)])
                h -= p_ans * np.log(p_ans)
            if h > max_h:
                max_h = h
                max_selection = selection
        # print("base: ", max_selection, "max_h: ", max_h)
        return np.array(max_selection), facts.get_subset(list(max_selection)), max_h


# 贪心法的问题选择器
class GreedyQuerySelector(QuerySelector):  # 改6
    """
    贪心法的问题选择器
    """

    def select(self, facts: FactSet,
               num: int,
               worker_accuracy: np.ndarray) -> Tuple[np.ndarray, "FactSet", float]:
        num_fact: int = facts.num_fact()
        if num > num_fact:  # k等于5，但fact只有4
            num = num_fact
        assert num <= num_fact
        max_selection: list = []
        # trt_selection: list = [] #不必要
        now_num = 0  # 避免计算len(max_selection)
        while now_num < num:  # 找到熵最大的fact组合
            max_h: float = -float('inf')  # 对每次的fact组合取该长度组合的熵最大
            max_idx = -1
            for idx in range(num_fact):
                # print(idx,end=":")
                if idx in max_selection:
                    continue
                max_selection.append(idx)
                sub_facts = facts.get_subset(list(max_selection))
                h = 0.
                # 计算answer set的墒
                # print(max_selection,end=":")
                for i in range(len(sub_facts)):
                    p_ans, _ = sub_facts.compute_ans_p(sub_facts[i],
                                                       list(range(sub_facts.num_fact())),
                                                       worker_accuracy[:, list(max_selection)])
                    h -= p_ans * np.log(p_ans)
                # print(h)
                if h > max_h:
                    max_h = h
                    max_idx = idx   # 每次找出最大的idx在循环外部append
                    # max_selection = copy.copy(max_selection)
                max_selection.pop(now_num)  # 每次删除的位置一定改用pop()会更快
            max_selection.append(max_idx)
            now_num += 1
        return np.array(max_selection), facts.get_subset(list(max_selection)), max_h


class RandomQuerySelector(QuerySelector):  #2.6
    """
    随机法的问题选择器
    """
    def select(self, facts: FactSet,
               num: int,
               worker_accuracy: np.ndarray
               ) -> Tuple[np.ndarray, "FactSet", float]:
        num_fact: int = facts.num_fact()
        if num > num_fact:
            num = num_fact
        assert num <= num_fact
        import random
        import numpy as np
        selection = random.sample(range(0, num_fact),num)
        # selection = np.random.choice(num_fact,num,replace=False)
        # sub_facts = facts.get_subset(list(selection))
        sub_facts = facts.get_subset(selection)
        h = 0
        for i in range(len(sub_facts)):
            p_ans, _ = sub_facts.compute_ans_p(sub_facts[i],
                                               list(range(sub_facts.num_fact())),
                                               worker_accuracy[:,list(selection)])
            h -= p_ans * np.log2(p_ans)
        # print("base: ", selection, "max_h: ", h)
        return np.array(selection), facts.get_subset(list(selection)), h