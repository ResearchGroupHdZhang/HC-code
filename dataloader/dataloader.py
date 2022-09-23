from os import P_WAIT
import numpy as np
from itertools import combinations
from fact import FactSet
from typing import Tuple, List
import abc
import itertools


class DataLoader(metaclass=abc.ABCMeta):
    """
    数据存储器
    """
    @abc.abstractmethod
    def __init__(self):
        self.idx = None
        self.prior_p = None
        self.ground_truth = None

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplemented("implement me")


class NormalDataloader(DataLoader):
    """
    正常迭代的数据存储器
    """
    def __init__(self, idx: List[np.ndarray],
                 prior_p: List[np.ndarray],
                 ground_truth: List[np.ndarray]):
        """
        :param idx: 数据的id
        :param prior_p: 数据的先验概率
        :param ground_truth: 数据的ground_truth
        """
        self.idx = idx
        self.prior_p = prior_p
        self.ground_truth = ground_truth
        self.primary_len = len(self.idx)
        self.dict_i = {}  # hwx:统计次数

    def __getitem__(self, item):
        """
        返回mean_entropy的方法为最简单的每轮计算一次
        """
        sum = 0
        time = 0
        for i, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            if time >= self.primary_len:
                break
            p = p.tolist()
            g = g.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
                """
                此处需拓展为计算先验概率的方法
                """
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] > 1e-5 else 1e-5, p[0] if p[0] >1e-5 else 1e-5])
            sum += np.sum(-p_t * np.log2(p_t)).item()
            time += 1
        return self.idx[item], self.prior_p[item], self.ground_truth[item], sum

    def add_data(self, idx, post_p, gt):
        self.idx.append(idx)
        self.prior_p[len(self.prior_p) % self.primary_len] = post_p
        self.prior_p.append(post_p)
        self.ground_truth.append(gt)

    def wirte_post_p_to_txt(self, txt_name, output_file):
        """
        写入最后状态(post_p)
        """
        with open(output_file + "output_post_p_txt/"+txt_name+".txt", "w") as f:
            for i in range(self.primary_len):
                f.write(str(self.idx[i].tolist()[0]))
                f.write('\n')
                p = self.prior_p[i].tolist()
                gt = self.ground_truth[i].tolist()
                for j in range(len(gt)):
                    if len(gt) == 1:
                        f.write(str(p[0]))
                        f.write(' ')
                        break
                    f.write(str(p[2**j]))
                    f.write(' ')
                f.write('\n')
                for j in gt:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')


class IncreEntropyDataloader(DataLoader):
    """
    按熵值迭代的数据存储器
    """
    def __init__(self, idx: List[np.ndarray],
                 prior_p: List[np.ndarray],
                 ground_truth: List[np.ndarray]):
        """
        :param idx: 数据的id
        :param prior_p: 数据的先验概率
        :param ground_truth: 数据的ground_truth
        :param entropy_list: 熵值从高到底的item的index
        """
        self.idx = idx
        self.prior_p = prior_p
        self.ground_truth = ground_truth
        self.entropy_list = None
    
    def get_subset(mylist, k):
        val = []
        for i in itertools.combinations(mylist, k):
            val.append(list(i))
        return val

    def __getitem__(self, item):
        """
        方法为最原始的每轮重排序选出熵值最大，顺便可以返回mean_entropy
        """
        tmp = [] # tmp = [h1 h2 h3 ... h722]
        sum = 0
        for i, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            p = p.tolist()
            g = g.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
                """
                此处需拓展为计算先验概率的方法
                """
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] != 0 else 1e-5, p[0]])
            tmp.append(np.sum(-p_t * np.log2(p_t)).item())
            sum += np.sum(-p_t * np.log2(p_t)).item()
        self.entropy_list = sorted(range(len(tmp)), key=lambda k: -tmp[k])
        return_i, return_p, return_g = self.idx[self.entropy_list[0]], self.prior_p[self.entropy_list[0]], self.ground_truth[self.entropy_list[0]]
        del self.idx[self.entropy_list[0]]
        del self.prior_p[self.entropy_list[0]]
        del self.ground_truth[self.entropy_list[0]]
        return return_i, return_p, return_g, sum

    def add_data(self, idx, post_p, gt):
        self.idx.append(idx)
        self.prior_p.append(post_p)
        self.ground_truth.append(gt)

    def wirte_post_p_to_txt(self, txt_name, output_file):
        """
        写入最后状态(post_p)
        """
        with open(output_file + "output_post_p_txt/"+txt_name+".txt", "w") as f:
            for i in range(len(self.idx)):
                f.write(str(self.idx[i].tolist()[0]))
                f.write('\n')
                p = self.prior_p[i].tolist()
                for j in p:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
                gt = self.ground_truth[i].tolist()
                for j in gt:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')

class BruteChoiceDataloader(DataLoader):
    """
    数据存储器——暴力法
    Task(n fact)的排序方式: Max ([Cn取k] 个fact的熵值) 为Task权重
    """
    def __init__(self, idx: List[np.ndarray],
                 prior_p: List[np.ndarray],
                 ground_truth: List[np.ndarray],
                 k: int):
        """
        :param idx: 数据的id
        :param prior_p: 数据的先验概率
        :param ground_truth: 数据的ground_truth
        :param entropy_list: 熵值从高到底的item的index
        """
        self.idx = idx
        self.prior_p = prior_p
        self.ground_truth = ground_truth
        self.entropy_list = None
        self.k = k
        self.dict_i = {}  # hwx:统计次数

    def get_subset(self, mylist, k):
        val = []
        for i in itertools.combinations(mylist, k):
            val.append(list(i))
        return val

    def __getitem__(self, item):
        tmp = [] # tmp = [task1权重 task2权重 ... task722权重]
        sum = 0 # sum 为总熵值
        for i, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            p = p.tolist()
            g = g.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] != 0 else 1e-5, p[0]])
            # 修改权重部分
            if p_t.size < self.k:
                tmp.append(np.sum(-p_t * np.log2(p_t)).item())
            else:
                # 所有可能的序列逐个遍历找到质量最好的
                combis = self.get_subset(range(p_t.size), k=self.k)
                max_combi_h = 0
                max_combi = []
                for combi in combis:
                    p_t_combi = p_t[np.array(combi)]
                    p_t_combi_h = np.sum(-p_t_combi * np.log2(p_t_combi)).item()
                    if p_t_combi_h > max_combi_h:
                        max_combi = p_t_combi
                        max_combi_h = p_t_combi_h
                tmp.append(max_combi_h)
            #
            sum += np.sum(-p_t * np.log2(p_t)).item()
        self.entropy_list = sorted(range(len(tmp)), key=lambda k: -tmp[k])
        # return_i, return_p, return_g = self.idx[self.entropy_list[0]], self.prior_p[self.entropy_list[0]], self.ground_truth[self.entropy_list[0]]
        # del self.idx[self.entropy_list[0]]
        # del self.prior_p[self.entropy_list[0]]
        # del self.ground_truth[self.entropy_list[0]]
        i = 0
        # print(self.idx[self.entropy_list[i]][0])
        flag = 1
        while flag == 1:
            if self.idx[self.entropy_list[i]][0] not in self.dict_i:
                self.dict_i[self.idx[self.entropy_list[i]][0]] = 1
                flag = 0
            elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= 7:
                i += 1
            else:
                self.dict_i[self.idx[self.entropy_list[i]][0]] += 1
                flag = 0

        return_i, return_p, return_g = self.idx[self.entropy_list[i]], self.prior_p[self.entropy_list[i]], \
                                       self.ground_truth[self.entropy_list[i]]
        # print(return_i)
        del self.idx[self.entropy_list[i]]
        del self.prior_p[self.entropy_list[i]]
        del self.ground_truth[self.entropy_list[i]]
        return return_i, return_p, return_g, sum

    def add_data(self, idx, post_p, gt):
        self.idx.append(idx)
        self.prior_p.append(post_p)
        self.ground_truth.append(gt)

    def wirte_post_p_to_txt(self, txt_name, output_file):
        """
        写入最后状态(post_p)
        """
        with open(output_file + "output_post_p_txt/"+txt_name+".txt", "w") as f:
            for i in range(len(self.idx)):
                f.write(str(self.idx[i].tolist()[0]))
                f.write('\n')
                p = self.prior_p[i].tolist()
                for j in p:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
                gt = self.ground_truth[i].tolist()
                for j in gt:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')


class ApproxChoiceDataloader(DataLoader):
    """
    数据存储器——近似法
    Task(n fact)的排序方式: Max ([Cn取k] 个fact的熵值) 为Task权重
    """
    def __init__(self, idx: List[np.ndarray],
                 prior_p: List[np.ndarray],
                 ground_truth: List[np.ndarray],
                 k: int):
        """
        :param idx: 数据的id
        :param prior_p: 数据的先验概率
        :param ground_truth: 数据的ground_truth
        :param entropy_list: 熵值从高到底的item的index
        """
        self.idx = idx
        self.prior_p = prior_p
        self.ground_truth = ground_truth
        self.entropy_list = None
        self.k = k
        self.dict_i = {} #hwx:统计次数

    def __getitem__(self, item):
        tmp = []  # tmp = [task1权重 task2权重 ... task722权重]
        sum = 0  # sum 为总熵值
        for i, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            p = p.tolist()
            g = g.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] != 0 else 1e-5, p[0]])
            # 修改权重部分
            if p_t.size < self.k:
                tmp.append(np.sum(-p_t * np.log2(p_t)).item())
            else:
                max_choice = []
                while len(max_choice) != self.k:
                    max_select_fact = 0
                    max_select_fact_h = 0
                    for index in range(p_t.size):
                        if index in max_choice:
                            continue
                        h_now = np.sum(-p_t[index] * np.log2(p_t[index])).item()
                        if h_now > max_select_fact_h:
                            max_select_fact_h = h_now
                            max_select_fact = index
                    max_choice.append(max_select_fact)
                tmp.append(np.sum(-p_t[np.array(max_choice)] * np.log2(p_t[np.array(max_choice)])).item())

            sum += np.sum(-p_t * np.log2(p_t)).item()
        self.entropy_list = sorted(range(len(tmp)), key=lambda k: -tmp[k])

        # hwx:统计次数,不超过3次
        i= 0
        # print(self.idx[self.entropy_list[i]][0])
        flag = 1
        while flag == 1:
            if self.idx[self.entropy_list[i]][0] not in self.dict_i:
                self.dict_i[self.idx[self.entropy_list[i]][0]] = 1
                flag = 0
            elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= 7:
                i += 1
            else:
                self.dict_i[self.idx[self.entropy_list[i]][0]] += 1
                flag = 0

        return_i, return_p, return_g = self.idx[self.entropy_list[i]], self.prior_p[self.entropy_list[i]], \
                                       self.ground_truth[self.entropy_list[i]]
        # print(return_i)
        del self.idx[self.entropy_list[i]]
        del self.prior_p[self.entropy_list[i]]
        del self.ground_truth[self.entropy_list[i]]

        return return_i, return_p, return_g, sum

    def add_data(self, idx, post_p, gt):
        self.idx.append(idx)
        self.prior_p.append(post_p)
        self.ground_truth.append(gt)

    def wirte_post_p_to_txt(self, txt_name, output_file):
        """
        写入最后状态(post_p)
        """
        with open(output_file + "output_post_p_txt/"+txt_name+".txt", "w") as f:
            for i in range(len(self.idx)):
                f.write(str(self.idx[i].tolist()[0]))
                # f.write(str(self.idx[i]))
                f.write('\n')
                p = self.prior_p[i].tolist()
                for j in p:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
                gt = self.ground_truth[i].tolist()
                for j in gt:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
