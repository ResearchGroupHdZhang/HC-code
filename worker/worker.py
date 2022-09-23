import string
import abc

import numpy as np
from typing import Sequence, Callable, List, Union, Tuple
from random import sample


class Worker:
    def __init__(self,
                 answer_samplers: Sequence["AnswerSampler"],
                 name: str):
        self._accuracy = np.array([s.get_p() for s in answer_samplers])
        self._samplers = answer_samplers
        self._name = name

    def name(self) -> str:
        return self._name

    def get_accuracy(self) -> np.ndarray:
        return self._accuracy

    def get_answer(self, query: Union[np.ndarray, List]) -> np.ndarray:
        """
        给定问题，返回答案
        :param query: 1D arrays，对应fact的索引
        :return: 对应的answer
        """
        return np.array([s.get_answer() for s in self._samplers])[query]


class AnswerSampler(metaclass=abc.ABCMeta):
    """
    单个fact的回答生成器的抽象类
    """

    @abc.abstractmethod
    def get_answer(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_p(self) -> float:
        raise NotImplementedError


class BaseAnswerSampler(AnswerSampler):

    def __init__(self,
                 fact_val: Union[Sequence[int], np.ndarray],
                 ground_true: int, p: float):
        """
        简单的回答生成器，按照概率p生成正确答案，否则随机选取答案
        :param fact_val: 该fact的候选值，e.g. [2,4,5,6]
        :param ground_true: 指示fact_val中真值的索引
        :param p: 该生成器采样为真值的概率
        """
        self._fact_val = fact_val
        self._ground_true = ground_true
        self._p = p
        self._ans_dist = np.ones_like(fact_val) * (1 - p) / (len(fact_val) - 1)
        self._ans_dist[ground_true] = p  # 各个答案的多项式分布

    def get_answer(self) -> int:
        return np.random.choice(self._fact_val, 1, p=self._ans_dist).item()

    def get_p(self) -> float:
        return self._p


class WorkerFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_worker(self) -> "Worker":
        pass


class BaseWorkerFactor(WorkerFactory):

    def __init__(self, ground_true_list: Union[Sequence, np.ndarray] = None,
                 accuracy_sampler: Callable[[], float] = None,
                 fact_space: List[List[int]] = None):
        super().__init__()
        self._ground_true = ground_true_list
        self._accuracy_sampler = accuracy_sampler
        if ground_true_list is not None:
            self._num_fact: int = len(ground_true_list)
        else:
            self._num_fact = None
        self._fact_space: List[List[int]] = fact_space

    def set_ground_true(self, ground_true: Union[Sequence, np.ndarray]):
        if self._ground_true is not None:
            assert len(self._ground_true) == len(ground_true)
        self._ground_true = ground_true
        self._num_fact = len(ground_true)

    def set_fact_space(self, fact_space: List[List[int]]):
        assert len(self._fact_space) == len(fact_space)
        self._fact_space = fact_space

    def get_worker(self, worker_acc) -> "Worker":
        # 修改为均值0.8,方差0.01的正态分布
        # worker_accuracy = [np.random.normal(self._accuracy_sampler(), 0.01) for _ in range(self._num_fact)]
        # 新
        # worker_acc = np.random.normal(self._accuracy_sampler(), 0.01)
        # #
        worker_accuracy = [worker_acc
                           for _ in range(self._num_fact)]
        # print(worker_accuracy)
        ans_samplers: List["AnswerSampler"] = [
            BaseAnswerSampler(space, gt, p)
            for space, gt, p in zip(self._fact_space,
                                    self._ground_true,
                                    worker_accuracy)
        ]
        name: str = "".join(sample(string.digits + string.ascii_letters, 5))
        worker: Worker = Worker(ans_samplers, name)
        return worker
