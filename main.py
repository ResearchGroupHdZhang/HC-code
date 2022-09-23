import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import math
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector, RandomQuerySelector
from fact import FactSet
from worker import Worker, WorkerFactory, BaseWorkerFactor
from crowsourcing_processor import BaseCrowdSourcingProcessor, CrowdSourcingProcessor, QuerySettingOption
from functools import lru_cache
from dataloader import *
from param import param

np.random.seed(5)  # 控制随机种子

@lru_cache(1)
def read_raw_data(dataname:str,method:str,deta:float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # with open("./different_algorithms/datasets/disease_to_symptom.txt", "r") as f:
    with open(input_file + dataname + '_Inputdata' + method + '_' + str(deta) + '.txt', "r") as f:
        raw_lines = f.readlines()
        
    idxes = []
    prior_p = []
    labels = []
    index = 0
    while index != len(raw_lines):
        idx = raw_lines[index]
        index += 1
        p = raw_lines[index].strip().split(" ")
        # p = raw_lines[index].strip().split("\t")
        p = [float(i) for i in p]
        p = np.asarray(p)
        index += 1
        l = raw_lines[index].strip().split(" ")
        # l = raw_lines[index].strip().split("\t")

        l = [int(i) for i in l]
        l = np.asarray(l)
        idxes.append(np.asarray([int(idx)]))
        prior_p.append(p)
        labels.append(l)
        index += 1
    return idxes, prior_p, labels


def get_query_selection_num() -> int:
    return 1


def get_budget() -> int:
    print("budget:",math.ceil(length / group_num) * kv )
    return math.ceil(length / group_num) * kv
    # return 6

def get_factset(split_idx: int) -> "FactSet":
    idxes, prior_p, labels = read_raw_data()
    p = (1 - labels[split_idx]) * np.array([1 - prior_p[split_idx], prior_p[split_idx]])
    p += (labels[split_idx]) * np.array([prior_p[split_idx], 1 - prior_p[split_idx]])
    return FactSet(
        facts=np.array([[0], [1]]),
        prior_p=p,
        ground_true=labels[0][split_idx],
    )


def get_processor(worker_factory: WorkerFactory, query_selector: QuerySelector,dataname:str,deta:float,low:float,theta:float) -> CrowdSourcingProcessor:
    processor = BaseCrowdSourcingProcessor(worker_factory, query_selector,
                                           get_query_selection_num())
    processor.setDataname(dataname, deta, low, theta)
    processor.budgets = get_budget()
    return processor


def start(kv:int, dataname:str, method:str, deta:float, selector_id:int, id:int):
    acc_ = param.acc_
    selector_id = selector_id
    # **zhenh** expert_num 不固定
    # expert_num = 2
    low = param.low
    theta = param.theta

    # 修改为正态分布的平均值
    def mean_dist() -> float:
        return acc_

    # 比较三种方法
    if id == 1:
        k_ = kv

        worker_factor = BaseWorkerFactor(accuracy_sampler=acc_, fact_space=[[0, 1]])
        if selector_id == 1:
            query_selector = BaseQuerySelector()
        elif selector_id == 2:
            query_selector = GreedyQuerySelector()
        else:
            query_selector = RandomQuerySelector()

        processor = get_processor(worker_factor, query_selector, dataname, deta, low, theta)
        w_labels = processor.getWlabels()
        a, b, c = read_raw_data(dataname, method, deta)

        if selector_id == 1:
            s_dataloader = BruteChoiceDataloader(a, b, c, k=k_)
        elif selector_id == 2:
            s_dataloader = ApproxChoiceDataloader(a, b, c, k=k_)
        else:
            s_dataloader = NormalDataloader(a, b, c)

        time = 0
        times = []
        mean_entropies = []

        for idx, prior_p, gt, mean_entropy in s_dataloader:
            # print(idx)
            times.append(time)
            mean_entropies.append(mean_entropy)
            print("method: " + method + " time: " + str(time) + " " + str(mean_entropy))
            prior_p = prior_p.tolist()
            gt = gt.tolist()
            # 若事实集只有一个
            if len(prior_p) == 1:
                k = 1
                # 若真实标签为1
                if gt[0] == 1:
                    factset = FactSet(np.array([[0], [1]]),
                                      prior_p=np.array([1 - prior_p[0], prior_p[0]]),
                                      ground_true=1,
                                      fact_space=[[0, 1]])
                    worker_factor.set_ground_true(np.array([1]))
                # yjy：若真实标签为0
                else:
                    factset = FactSet(np.array([[0], [1]]),
                                      prior_p=np.array([1 - prior_p[0], prior_p[0]]),
                                      ground_true=0,
                                      fact_space=[[0, 1]])
                    worker_factor.set_ground_true(np.array([0]))
                processor.change_worker_factory(worker_factor)
                Q_op = QuerySettingOption(k)
                Q_op.set(processor)
            # 事实集有多个
            elif len(prior_p) == len(gt):
                k = k_
                factset_len = 2 ** len(gt)
                tmp_factset = []
                tmp_factspace = []
                s_format = '{:0' + str(len(gt)) + 'b}'
                # yjy：通过转成二进制获得所有facts排列组合
                for i in range(factset_len):
                    s = s_format.format(i)
                    s = list(s)
                    tmp = []
                    for j in range(len(s)):
                        tmp.append(int(s[j]))
                    tmp_factset.append(tmp)
                # yjy：每个事实的可能取值 f1取[0,1], f2取[0,1], f3取[0,1]....
                for i in range(factset_len):
                    tmp_factspace.append([0, 1])
                worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
                gt_str = [str(i) for i in gt]
                gt_str.reverse()
                gt_str = ''.join(gt_str)
                ground_true_int = int(gt_str, 2)
                tmp_prior_p = [1e-5 for i in range(2 ** len(prior_p))]
                for i in range(len(prior_p)):
                    tmp_prior_p[2 ** i] = prior_p[i]
                factset = FactSet(np.array(tmp_factset),
                                  prior_p=np.array(tmp_prior_p),
                                  ground_true=ground_true_int,
                                  fact_space=tmp_factspace
                                  )
                # if time == 0:
                #     print(tmp_factset)
                #     test = pd.DataFrame(data=tmp_factset)
                #     test.to_csv('./different_algorithms/datasets/factset10.csv', index=False, encoding='gbk')

                gt.reverse()
                worker_factor1.set_ground_true(np.array(gt))
                processor.change_worker_factory(worker_factor1)
                Q_op = QuerySettingOption(k)
                Q_op.set(processor)
            else:
                k = k_
                factset_len = len(prior_p)
                tmp_factset = []
                tmp_factspace = []
                s_format = '{:0' + str(len(gt)) + 'b}'
                for i in range(factset_len):
                    s = s_format.format(i)
                    s = list(s)
                    tmp = []
                    for j in range(len(s)):
                        tmp.append(int(s[j]))
                    tmp_factset.append(tmp)
                for i in range(factset_len):
                    tmp_factspace.append([0, 1])
                worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
                gt_str = [str(i) for i in gt]
                gt_str.reverse()
                gt_str = ''.join(gt_str)
                ground_true_int = int(gt_str, 2)  # 001101-->13
                # print(tmp_factset)
                factset = FactSet(np.array(tmp_factset),
                                  prior_p=np.array(prior_p),
                                  ground_true=ground_true_int,
                                  fact_space=tmp_factspace
                                  )
                gt.reverse()
                worker_factor1.set_ground_true(np.array(gt))
                processor.change_worker_factory(worker_factor1)
                Q_op = QuerySettingOption(k)
                Q_op.set(processor)
            processor.init_task(factset)
            # **zhenh** expert_num 不固定
            processor.start_task(k, idx)
            post_p = factset.get_prior_p()

            # hwx:注释掉就不放回了
            if len(prior_p) == 1:
                s_dataloader.add_data(idx, np.asarray([post_p.tolist()[0]]), np.asarray(gt))
                # approx_dataloader.add_data(idx, np.asarray([post_p.tolist()[0]]), np.asarray(gt))
            else:
                s_dataloader.add_data(idx, post_p, np.asarray(gt))
                # approx_dataloader.add_data(idx, post_p, np.asarray(gt))

            time += len(w_labels[str(idx[0])])
            # 只要EBCC的
            if processor.is_end or method == 'MV':
                break


        # 只要EBCC基线的
        if method == param.method:
        # if method == 'DS':
        # if method == 'EBCC':
            # 写入处理后原数据的post_p(txt)
            if selector_id == 1:
                s_dataloader.wirte_post_p_to_txt(dataname + '_' + str(acc_) + 'acc+brute(k=' + str(k_) + ')' + method + '_compare', output_file)
                # s_dataloader.wirte_post_p_to_txt('test')
            elif selector_id == 2:
                s_dataloader.wirte_post_p_to_txt(dataname + '_' + str(acc_) + 'acc+approx(k=' + str(k_) + ')' + method + '_compare', output_file)
            else:
                s_dataloader.wirte_post_p_to_txt(dataname + '_' + str(acc_) + 'acc+random(k=' + str(k_) + ')' + method + '_compare', output_file)
            # 写入budget-mean_entropy的csv
            dataframe = pd.DataFrame({'time': times, 'mean_entropy': mean_entropies})
            if selector_id == 1:
                # dataframe.to_csv("./outputcsv/test", index=Falase, sep=',')
                dataframe.to_csv(output_file + '/outputcsv/three_method/' + dataname + "_" + str(acc_) + "acc+brute(k=" + str(k_) + ")" + method + "_compare.csv", index=False, sep=',')
            elif selector_id == 2:
                dataframe.to_csv(output_file + '/outputcsv/three_method/' + dataname + "_" + str(acc_) + "acc+approx(k=" + str(k_) + ")" + method + "_compare.csv", index=False, sep=',')
            else:
                dataframe.to_csv(output_file + '/outputcsv/three_method/' + dataname + "_" + str(acc_) + "acc+random(k=" + str(k_) + ")" + method + "_compare.csv", index=False, sep=',')

            processor.outputcsv(kv=kv, method=method, output_file=output_file, k=k_)

    # --------------------------------------------------------------------------------------------
    # 比较不同初始化与不同k时 selector_id = 2 且k_ in (1, 7)
    else:
        # 任务选择数量k_
        for k_ in range(1, group_num+1):
            # for method in ['EBCC','MV']:
            for method in ['MV', 'EBCC', 'DS', 'GLAD', 'BCC', 'BWA', 'ZC', 'PM']:
                print('************:', method, kv, k_)
                worker_factor = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=[[0, 1]])
                query_selector = GreedyQuerySelector()

                processor = get_processor(worker_factor, query_selector, dataname, deta, low, theta)
                a, b, c = read_raw_data(dataname, method, deta)

                s_dataloader = ApproxChoiceDataloader(a, b, c, k=k_)
                # s_dataloader = NormalDataloader(a, b, c)

                time = 0
                w_labels = processor.getWlabels()
                times = []
                mean_entropies = []
                count = 0

                idxs = []
                ps = []
                gts = []

                # for idx, prior_p, gt, mean_entropy in approx_dataloader:
                for idx, prior_p, gt, mean_entropy in s_dataloader:
                    count+=1
                    # print("count:",count)
                    times.append(time)
                    mean_entropies.append(mean_entropy)
                    print("method: " + method + " time: " + str(time) + " " + str(mean_entropy))
                    prior_p = prior_p.tolist()
                    gt = gt.tolist()
                    # yjy：若事实集只有一个
                    if len(prior_p) == 1:
                        k = 1
                        # yjy：若真实标签为1
                        if gt[0] == 1:
                            factset = FactSet(np.array([[0], [1]]),
                                              prior_p=np.array([1 - prior_p[0], prior_p[0]]),
                                              ground_true=1,
                                              fact_space=[[0, 1]])
                            worker_factor.set_ground_true(np.array([1]))
                        # yjy：若真实标签为0
                        else:
                            factset = FactSet(np.array([[0], [1]]),
                                              prior_p=np.array([1 - prior_p[0], prior_p[0]]),
                                              ground_true=0,
                                              fact_space=[[0, 1]])
                            worker_factor.set_ground_true(np.array([0]))
                        processor.change_worker_factory(worker_factor)
                        Q_op = QuerySettingOption(k)
                        Q_op.set(processor)
                    # 有些数据prior_p数量与gt相同
                    elif len(prior_p) == len(gt):
                        k = k_
                        factset_len = 2 ** len(gt)
                        tmp_factset = []
                        tmp_factspace = []
                        s_format = '{:0' + str(len(gt)) + 'b}'
                        # yjy：通过转成二进制获得所有facts排列组合
                        for i in range(factset_len):
                            s = s_format.format(i)
                            s = list(s)
                            tmp = []
                            for j in range(len(s)):
                                tmp.append(int(s[j]))
                            tmp_factset.append(tmp)
                        # yjy：每个事实的可能取值 f1取[0,1], f2取[0,1], f3取[0,1]....
                        for i in range(factset_len):
                            tmp_factspace.append([0, 1])
                        worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
                        gt_str = [str(i) for i in gt]
                        gt_str.reverse()
                        gt_str = ''.join(gt_str)
                        ground_true_int = int(gt_str, 2)
                        tmp_prior_p = [1e-5 for i in range(2 ** len(prior_p))]
                        for i in range(len(prior_p)):
                            tmp_prior_p[2 ** i] = prior_p[i]
                        factset = FactSet(np.array(tmp_factset),
                                          prior_p=np.array(tmp_prior_p),
                                          ground_true=ground_true_int,
                                          fact_space=tmp_factspace
                                          )
                        # if time == 0:
                        #     print(tmp_factset)
                        #     test = pd.DataFrame(data=tmp_factset)
                        #     test.to_csv('./different_algorithms/datasets/factset10.csv', index=False, encoding='gbk')

                        gt.reverse()
                        worker_factor1.set_ground_true(np.array(gt))
                        processor.change_worker_factory(worker_factor1)
                        Q_op = QuerySettingOption(k)
                        Q_op.set(processor)
                    else:
                        k = k_
                        factset_len = len(prior_p)
                        tmp_factset = []
                        tmp_factspace = []
                        s_format = '{:0' + str(len(gt)) + 'b}'
                        for i in range(factset_len):
                            s = s_format.format(i)
                            s = list(s)
                            tmp = []
                            for j in range(len(s)):
                                tmp.append(int(s[j]))
                            tmp_factset.append(tmp)
                        for i in range(factset_len):
                            tmp_factspace.append([0, 1])
                        worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
                        gt_str = [str(i) for i in gt]
                        gt_str.reverse()
                        gt_str = ''.join(gt_str)
                        ground_true_int = int(gt_str, 2)
                        # print(tmp_factset)
                        factset = FactSet(np.array(tmp_factset),
                                          prior_p=np.array(prior_p),
                                          ground_true=ground_true_int,
                                          fact_space=tmp_factspace
                                          )
                        gt.reverse()
                        worker_factor1.set_ground_true(np.array(gt))
                        processor.change_worker_factory(worker_factor1)
                        Q_op = QuerySettingOption(k)
                        Q_op.set(processor)
                    processor.init_task(factset)
                    # **zhenh** expert_num 不固定
                    # processor.start_task(B, k, expert_num, idx)
                    processor.start_task(k, idx)
                    post_p = factset.get_prior_p()

                    #hwx:注释掉就不放回了
                    # if count % 200 == 0:
                    if len(prior_p) == 1:
                        s_dataloader.add_data(idx, np.asarray([post_p.tolist()[0]]), np.asarray(gt))
                    else:
                        s_dataloader.add_data(idx, post_p, np.asarray(gt))
                        # for idx, post_p, gt in zip(idxs,ps,gts):
                        #     s_dataloader.add_data(idx, post_p, gt)
                            # idxs = []
                            # ps = []
                            # gts = []
                    # else:
                    #     idxs.append(idx)
                    #     ps.append(post_p)
                    #     gts.append(np.asarray(gt))


                    time += len(w_labels[str(idx[0])])

                    # test: MV不用打印
                    # if processor.is_end or method=='MV':
                    #     break

                    # 正式实验
                    if processor.is_end:
                        break

                # 写入处理后原数据的post_p(txt)
                s_dataloader.wirte_post_p_to_txt(dataname + '_' + str(acc_) + 'acc+approx(k=' + str(kv) + ')' + method + '_k=' + str(k_), output_file)

                # 写入budget-mean_entropy的csv
                dataframe = pd.DataFrame({'time': times, 'mean_entropy': mean_entropies})
                dataframe.to_csv(output_file + '/outputcsv/diff_init/' + dataname + "_" + str(acc_) + "acc+approx(k=" + str(kv) + ")" + method + "_k=" + str(k_) + ".csv", index=False, sep=',')

                if method == param.method:
                # if method == 'DS':
                # if method == 'EBCC':
                    # 输出d_sentiment_exworker_select(k=kv)method.csv
                    processor.outputcsv(kv=kv, method=method, output_file=output_file, k=k_)
                    # 写入budget-mean_entropy的csv (diff_k)
                    dataframe.to_csv(output_file + '/outputcsv/diff_k/' + dataname + "_" + str(acc_) + "acc+approx(k=" + str(kv) + ")" + method + "_k=" + str(k_) + ".csv", index=False, sep=',')


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    source = './dataset/'

    deta = param.deta
    low = param.low
    dataname = param.dataname
    group_num = param.group_num
    # 获取数据长度
    length = param.get_length(source)

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/Transfer_initial2inputdata/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/'

    method = param.method

    # 比较三种不同方法的
    for selector_id in range(1, 4):
        for kv in range(1, 3):
            # 如果没有切换方法熵值不会重置
            for method in ['MV', param.method]:
                print('***********************:', 'selector_id=', selector_id, method, kv)
                start(kv, dataname, method, deta, selector_id, id=1)

    # approx k=4、5 另外计算
    for selector_id in range(2, 3):
        for kv in range(4, 6):
            # 如果没有切换方法熵值不会重置
            for method in ['MV', param.method]:
                print('***********************:', 'selector_id=', selector_id, method, kv)
                start(kv, dataname, method, deta, selector_id, id=1)

    # 比较不同初始化与不同k
    # kv表示有多少份预算
    for kv in range(1, group_num+2):
        # kv = 1 #hwx:从1改到5
        print('***********************:', method, kv)
        start(kv, dataname, method, deta, selector_id=2, id=2)




