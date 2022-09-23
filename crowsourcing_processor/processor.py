import logging
from typing import Any
import pandas as pd
import numpy as np

from worker import Worker, WorkerFactory
from query import QuerySelector
from fact import FactSet
import abc
from param import param

dataname = param.dataname

class CrowdSourcingProcessor(metaclass=abc.ABCMeta):
    def __init__(self):
        self.budgets: int = -1
        self.query_selector: QuerySelector = None
        self.factset: FactSet = None
        self.is_end = False

    @abc.abstractmethod
    def init_task(self, factset: FactSet):
        pass

    @abc.abstractmethod
    def start_task(self, ):
        pass


class BaseCrowdSourcingProcessor(CrowdSourcingProcessor):
    def __init__(self,
                 worker_factory: 'WorkerFactory' = None,
                 query_selector: 'QuerySelector' = None,
                 query_selection_num: int = -1):
        super().__init__()
        self.worker_factory = worker_factory
        self.query_selector = query_selector
        self.query_per_iteration = query_selection_num
        self.is_end = False  # is_end 修改：用于判断budgets用完
        self.acc_dic = {}
        # self.ans_dic = {}

        # **zhenh** 修改成列表
        # self.w1_label = {}
        # self.w2_label = {}
        # w_labels : { idx1: [ [], [] ], idx2: [ [], [], [] ], ... }
        # key为fact_id, value为列表，列表元素为专家的回答(字典)
        # 原先的 w_label: { idx: 回答 } 有多个w_label代表多个专家
        # 现在的 w_label: { idx: [[专家1回答], [专家2回答], ...]}
        self.w_labels = {}
        
        self.dataname = None
        self.deta = 0.0
        self.theta = 0.0
        self.idx = []
        self.query_idx = []

    def setDataname(self, dataname1, deta1, low1, theta1):
        self.dataname = dataname1
        self.deta = deta1
        # ------------------------
        self.low = low1
        self.theta = theta1
        # ------------------------

        # with open('./different_algorithms/datasets/worker_arr_test_10fact.txt', 'r') as f:
        with open('./dataset/' + self.dataname + '_' + str(self.deta) + '_' + str(self.low) + '/Preliminary_processing_data1_Output/' + self.dataname + '_worker_acc'+str(self.deta)+'_'+str(self.low)+ '_theta' + str(self.theta) +'.txt', 'r') as f:
            raw_lines = f.readlines()
            i = 0
            while i != len(raw_lines):
                idx = raw_lines[i].replace('\n', '')
                i += 1
                p = raw_lines[i].strip().split(' ')
                self.acc_dic[idx] = p
                i += 1
                # **zhenh** 修改成列表
                answer_list = []
                for j in range(len(p)):
                    answer_list.append(raw_lines[i].strip().split(' '))
                    i += 1
                self.w_labels[idx] = answer_list

    def getWlabels(self):
        return self.w_labels

    def init_task(self, factset: FactSet):
        super().init_task(factset)
        assert self.worker_factory is not None
        assert self.query_selector is not None
        assert self.query_per_iteration > 0
        self.factset = factset

    def start_task(self, k, id):
        # k : 选取了k个fact
        assert k != -1
        # print("budget rest:",self.budgets)
        if self.budgets >= k:
            ans_p_post_o_list = [] # list of p(ans|o), such as [p(ans1|o) p(ans2|o)] 
            o_prior_p = self.factset.get_prior_p()  # p(o)
            worker_list = []
            acc_list = []
            worker_accuracy = self.acc_dic[str(id[0])]
            worker_accuracy = [float(i) for i in worker_accuracy]
            # **zhenh** expert_num不固定
            expert_num = len(self.w_labels[str(id[0])])
            # yjy：创建专家序列和它对应的acc序列
            for i in range(expert_num):
                worker = self.worker_factory.get_worker(worker_accuracy[i])
                accuracy = worker.get_accuracy()
                worker_list.append(worker)
                acc_list.append(accuracy)
            
            acc_list = np.array(acc_list)
            # yjy：返回选择问题集的问题索引 问题子集 和对应的问题熵值
            query_idxes, sub_factset, ans_set_entropy = \
                    self.query_selector.select(self.factset,
                                               self.query_per_iteration,
                                               acc_list)
            # yjy：记录专家id 问题q_id
            for q_id in query_idxes:
                self.idx.append(int(id))
                self.query_idx.append(q_id)

            # yjy：0号专家
            for expert_index in range(expert_num):
                # ans = worker_list[expert_index].get_answer(query_idxes)
                ans_list = []
                for idx in query_idxes:
                    ans_list.append(int(self.w_labels[str(id[0])][expert_index][idx]))
                ans = np.array(ans_list)
                    
                # ans 分别是两个专家做出的回答
                ans_p, ans_p_post_o = self.factset.compute_ans_p(  # p(ans), p(ans|o)
                    ans, query_idxes, acc_list)
                ans_p_post_o_list.append(ans_p_post_o)
                self.budgets -= k
                if self.budgets < k:
                    break
            # yjy: 计算p(ans) 和 p(o|ans) = p(o)*p(ans|o) / p(ans)
            Cumprod = np.ones_like(ans_p_post_o_list[0])
            for i in ans_p_post_o_list:
                Cumprod = Cumprod * i
            o_p_post_ans = o_prior_p * Cumprod / (o_prior_p * Cumprod).sum()
            # yjy: 更新先验概率
            self.factset.set_prior_p(o_p_post_ans)
            #
        else:
            self.is_end = True

    def outputcsv(self,kv,method,output_file,k):
        dataframe = pd.DataFrame({'id': self.idx, 'query_idx': self.query_idx})
        dataframe.to_csv(output_file + 'exworker_select/' + dataname + '_exworker_select(k='+str(kv)+')'+method+'_'+str(self.deta)+'(k='+str(k)+')'+'.csv', index=False, sep=',')

    @staticmethod
    def from_options(*options: 'ProcessorOption'
                     ) -> 'BaseCrowdSourcingProcessor':
        p = BaseCrowdSourcingProcessor()
        for o in options:
            p = o.set(p)
        return p

    def change_worker_factory(self, new_worker_factory):
        self.worker_factory = new_worker_factory


class ProcessorOption(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set(self, processor: CrowdSourcingProcessor) -> 'CrowdSourcingProcessor':
        pass


class QuerySettingOption(ProcessorOption):

    def __init__(self, query_selection_num: int):
        assert query_selection_num > 0
        self._query_selection_num = query_selection_num

    def set(self, processor: BaseCrowdSourcingProcessor) -> 'CrowdSourcingProcessor':
        processor.query_per_iteration = self._query_selection_num
        return processor


class ProcessorResult(metaclass=abc.ABCMeta):
    '''
    表示CrowSourcingProcessor的任务运行结果
    '''

    @abc.abstractmethod
    def get_all(self) -> dict:
        pass

    @abc.abstractmethod
    def get(self, key) -> Any:
        pass
