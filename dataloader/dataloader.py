from os import P_WAIT
import numpy as np
from itertools import combinations
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector, RandomQuerySelector
from worker import Worker, WorkerFactory
from fact import FactSet
from typing import Tuple, List
import abc
import itertools
from param import param
import pandas as pd

source = './dataset/'


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
                 k: int,):
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
        self.k = k # 选择任务的数量
        self.dict_i = {}  # hwx:统计次数
        self.query_per_iteration = 1
        self.acc_list = []
        self.topic = {}
        self.w_labels = {}
        self.acc_dic = {}


    def get_topic(self):
        df = pd.read_csv('./dataset/' + param.dataname + '_' + str(param.deta) + '_' + str(param.low) + '_theta' + str(
            param.theta) + '/Preliminary_processing_data1_Output/' + param.dataname + '_id_topic.csv')
        id_list = df['id'].tolist()
        topic_list = df['topic'].tolist()
        for i in range(len(id_list)):
            self.topic[str(id_list[i])] = topic_list[i]
        return self.topic


    def get_acclist(self, id):
        with open('./dataset/' + param.dataname + '_' + str(param.deta) + '_' + str(param.low) + '_theta' + str(param.theta) + '/Preliminary_processing_data1_Output/' + param.dataname + '_worker_acc'+str(param.deta)+'_'+str(param.low)+ '_theta' + str(param.theta) +'.txt', 'r') as f:
            raw_lines = f.readlines()
            i = 0
            while i != len(raw_lines):
                idx = raw_lines[i].replace('\n', '')
                i += 1
                p = raw_lines[i].strip().strip('[').strip(']').strip(',').split('] [')
                p_list = []
                for p_topic in p:
                    p_topic_i = p_topic.split(', ')
                    p_list.append(p_topic_i)

                self.acc_dic[idx] = p_list
                i += 1
                # 修改成列表
                answer_list = []
                for j in range(len(p)):
                    answer_list.append(raw_lines[i].strip().split(' '))
                    i += 1
                self.w_labels[idx] = answer_list

        expert_num = len(self.w_labels[str(id[0])])
        f_topic = []
        worker_accuracy_topic = []
        worker_accuracy = self.acc_dic[str(id[0])]
        for acc_topic in worker_accuracy:
            worker_accuracy_topic.append([float(i) for i in acc_topic])
        for i in range(param.group_num):
            if id[0] + i >= param.get_length(source):
                break
            f_topic.append(self.topic[str(id[0] + i)])
        for i in range(expert_num):
            worker_acc = []
            for topic in f_topic:
                worker_acc.append(worker_accuracy_topic[i][topic])
        self.acc_list = np.array(worker_acc)
        self.acc_list = np.expand_dims(self.acc_list, axis=0)


    def get_subset(self, mylist, k):
        val = []
        for i in itertools.combinations(mylist, k):
            val.append(list(i))
        return val


    def __getitem__(self, item):
        tmp = [] # tmp = [task1权重 task2权重 ... task722权重]
        sum = 0 # sum 为总熵值
        topic = self.get_topic()
        for ix, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            # print('ix:' , ix)
            p = p.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] != 0 else 1e-5, p[0]])
            # 所有可能的序列逐个遍历找到质量最好的
            # change--------------------------------------------------
            tmp_factset = []
            tmp_factspace = []
            s_format = '{:0' + str(len(g)) + 'b}'
            factset_len = len(p_t)
            for i in range(factset_len):
                s = s_format.format(i)
                s = list(s)
                tmp_in = []
                for j in range(len(s)):
                    tmp_in.append(int(s[j]))
                tmp_factset.append(tmp_in)
            for i in range(factset_len):
                tmp_factspace.append([0, 1])


            gt_str = [str(i) for i in g]
            gt_str.reverse()
            gt_str = ''.join(gt_str)
            # print(gt_str)
            ground_true_int = int(gt_str, 2)
            facts = FactSet(np.array(tmp_factset),
                              prior_p=np.array(p_t),
                              ground_true=ground_true_int,
                              fact_space=tmp_factspace
                              )
            query_selector = BaseQuerySelector()
            self.get_acclist(ix)
            # print('self.acc_list', self.acc_list)
            query_idxes, _, max_h = query_selector.select(facts,
                                           self.k,
                                           self.acc_list)
            # print('query_idxes: ',query_idxes)
            tmp.append(max_h)

            sum += np.sum(-p_t * np.log2(p_t)).item()
            # tmp.append(np.sum(-p_t * np.log2(p_t)).item())
        print('tmp: ', tmp)
        self.entropy_list = sorted(range(len(tmp)), key=lambda a: -tmp[a])
        print('self.entropy_list: ', self.entropy_list)
        # print('idx: ', self.idx)
        # print('k, len(entropy_list):', self.k, len(self.entropy_list))

        i = 0
        flag = 1
        while flag == 1:
            if self.idx[self.entropy_list[i]][0] not in self.dict_i:
                self.dict_i[self.idx[self.entropy_list[i]][0]] = 1
                flag = 0
            # yjy:控制挑选次数
            elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= (param.tim * param.group_num) or self.dict_i[
                self.idx[self.entropy_list[i]][0]] - min(self.dict_i.values()) >= (param.tim * param.group_num // 2+(param.group_num-self.k)):
            # elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= (param.tim * param.group_num) or self.dict_i[
            #     self.idx[self.entropy_list[i]][0]] - min(self.dict_i.values()) > 2:
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
        print('dict_i: ', self.dict_i)
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
        :param prior_p: 数据的先验概率(2^n)
        :param ground_truth: 数据的ground_truth
        :param entropy_list: 熵值从高到底的item的index
        """
        self.idx = idx
        self.prior_p = prior_p
        self.ground_truth = ground_truth

        self.idx1 = idx
        self.prior_p1 = prior_p
        self.ground_truth1 = ground_truth

        self.entropy_list = None
        self.k = k
        self.dict_i = {} #hwx:统计次数
        self.query_per_iteration = 1
        self.acc_list = []
        self.topic = {}
        self.w_labels = {}
        self.acc_dic = {}


    def get_topic(self):
        df = pd.read_csv('./dataset/' + param.dataname + '_' + str(param.deta) + '_' + str(param.low) + '_theta' + str(
            param.theta) + '/Preliminary_processing_data1_Output/' + param.dataname + '_id_topic.csv')
        id_list = df['id'].tolist()
        topic_list = df['topic'].tolist()
        for i in range(len(id_list)):
            self.topic[str(id_list[i])] = topic_list[i]
        return self.topic


    def get_acclist(self, id):
        with open('./dataset/' + param.dataname + '_' + str(param.deta) + '_' + str(param.low) + '_theta' + str(param.theta) + '/Preliminary_processing_data1_Output/' + param.dataname + '_worker_acc'+str(param.deta)+'_'+str(param.low)+ '_theta' + str(param.theta) +'.txt', 'r') as f:
            raw_lines = f.readlines()
            i = 0
            while i != len(raw_lines):
                idx = raw_lines[i].replace('\n', '')
                i += 1
                p = raw_lines[i].strip().strip('[').strip(']').strip(',').split('] [')
                p_list = []
                for p_topic in p:
                    p_topic_i = p_topic.split(', ')
                    p_list.append(p_topic_i)

                self.acc_dic[idx] = p_list
                i += 1
                # 修改成列表
                answer_list = []
                for j in range(len(p)):
                    answer_list.append(raw_lines[i].strip().split(' '))
                    i += 1
                self.w_labels[idx] = answer_list
        expert_num = len(self.w_labels[str(id[0])])

        f_topic = []
        worker_accuracy_topic = []
        worker_accuracy = self.acc_dic[str(id[0])]
        for acc_topic in worker_accuracy:
            worker_accuracy_topic.append([float(i) for i in acc_topic])
        for i in range(param.group_num):
            if id[0] + i >= param.get_length(source):
                break
            f_topic.append(self.topic[str(id[0] + i)])
        for i in range(expert_num):
            worker_acc = []
            for topic in f_topic:
                worker_acc.append(worker_accuracy_topic[i][topic])
        self.acc_list = np.array(worker_acc)
        self.acc_list = np.expand_dims(self.acc_list, axis=0)


    def get_facset(self, fact):
        self.facts = fact


    def __getitem__(self, item):
        tmp = []  # tmp = [task1权重 task2权重 ... task722权重]
        sum_h = 0  # sum 为总熵值
        topic = self.get_topic()
        for ix, p, g in zip(self.idx, self.prior_p, self.ground_truth):
            p = p.tolist()
            g = g.tolist()
            if len(p) > 1:
                p = [i if i-0 > 1e-5 else 1e-5 for i in p]
                p_t = np.array(p)
            else:
                p_t = np.array([1 - p[0] if 1 - p[0] != 0 else 1e-5, p[0]])

            # 修改权重部分
            # change--------------------------------------------------
            tmp_factset = []
            tmp_factspace = []
            s_format = '{:0' + str(len(g)) + 'b}'
            factset_len = len(p_t)
            for i in range(factset_len):
                s = s_format.format(i)
                s = list(s)
                tmp_in = []
                for j in range(len(s)):
                    tmp_in.append(int(s[j]))
                tmp_factset.append(tmp_in)
            for i in range(factset_len):
                tmp_factspace.append([0, 1])

            gt_str = [str(i) for i in g]
            gt_str.reverse()
            gt_str = ''.join(gt_str)
            # print(gt_str)
            ground_true_int = int(gt_str, 2)
            self.facts = FactSet(np.array(tmp_factset),
                            prior_p=np.array(p_t),
                            ground_true=ground_true_int,
                            fact_space=tmp_factspace
                            )
            query_selector = GreedyQuerySelector()
            self.get_acclist(ix)
            # print('self.acc_list', self.acc_list)

            # --------------------------------------------------

            query_idxes, _, max_h = query_selector.select(self.facts,
                                                          self.k,
                                                          self.acc_list)
            # print('query_idxes: ', query_idxes)
            # print('max_h: ', max_h)
            tmp.append(max_h)

            sum_h += np.sum(-p_t * np.log2(p_t)).item()

        print('tmp:', tmp)
        # print('len(tmp): ', len(tmp))
        self.entropy_list = sorted(range(len(tmp)), key=lambda a: -tmp[a])
        print('self.entropy_list: ', self.entropy_list)
        # print('k, len(entropy_list):', self.k, len(self.entropy_list))

        # hwx:统计次数,不超过3次
        if self.k >=3:
            max_sel = 1
        else:
            max_sel = (param.tim * param.group_num // 2+(param.group_num-self.k))
        i= 0
        print(self.idx[self.entropy_list[i]][0])
        flag = 1
        while flag == 1:
            if self.idx[self.entropy_list[i]][0] not in self.dict_i:
                self.dict_i[self.idx[self.entropy_list[i]][0]] = 1
                flag = 0
            # yjy:控制挑选次数
            # elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= (param.tim * param.group_num) or self.dict_i[
            #     self.idx[self.entropy_list[i]][0]] - min(self.dict_i.values()) >= (param.tim * param.group_num // 2+(param.group_num-self.k)):
            elif self.dict_i[self.idx[self.entropy_list[i]][0]] >= (param.tim * param.group_num) or self.dict_i[
                self.idx[self.entropy_list[i]][0]] - min(self.dict_i.values()) >= max_sel:
                i += 1
            else:
                self.dict_i[self.idx[self.entropy_list[i]][0]] += 1
                flag = 0

        return_i, return_p, return_g = self.idx[self.entropy_list[i]], self.prior_p[self.entropy_list[i]], \
                                       self.ground_truth[self.entropy_list[i]]


        del self.idx[self.entropy_list[i]]
        del self.prior_p[self.entropy_list[i]]
        del self.ground_truth[self.entropy_list[i]]
        print('dict_i: ', self.dict_i)

        return return_i, return_p, return_g, sum_h

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
