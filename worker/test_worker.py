import unittest
from worker import BaseAnswerSampler, Worker
import logging
import numpy as np


class TestBaseAnswerSampler(unittest.TestCase):

    def test_get_answer(self):
        gt = 2
        num_sample = 1000
        p = 0.7
        sampler = BaseAnswerSampler([0, 1, 2, 3], gt, p)
        cnt_true = 0
        for i in range(num_sample):
            if sampler.get_answer() == 2:
                cnt_true += 1
        real_p = cnt_true / num_sample
        logging.info(f"p = {p}, real_p = {real_p}")
        self.assertLess(abs(real_p - p), 0.05, f"p={cnt_true / num_sample}")


class TestWorker(unittest.TestCase):
    def test_get_answer(self):
        # 3个fact，每个都有4个值，每个fact的真值假设为索引对应2的那个
        gt = 2
        num_sample = 1000
        num_fact = 3
        samplers = []
        for i, p in zip(range(num_fact), [0.6, 0.7, 0.8]):
            sampler = BaseAnswerSampler([0, 1, 2, 3], gt, p)
            samplers.append(sampler)
        worker = Worker(answer_samplers=samplers, name="114514")
        cnt_true = [0, 0, 0]
        for i in range(num_sample):
            ans = worker.get_answer(query=[0, 1, 2])  # 回答所有fact(3个)
            for i, is_true in enumerate(ans == gt):
                if is_true:
                    cnt_true[i] += 1
        for p, p_real in zip([0.6, 0.7, 0.8], np.array(cnt_true) / num_sample):
            self.assertLess(abs(p - p_real), 0.05, f"p:{p}, p_real:{p_real}")
