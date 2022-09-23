import pandas as pd
deta = 0.85 # hwx : deta和low其实一样的，不用做0.8-0.9的话。调了data,记得数据文件可能也需要改噢
low = 0.9
# 分组数量
group_num = 5

# **zhenh** theta
theta = 0.9
# **zhenh** fact条数
fact_num = 1000

# 基线算法
# method = 'DS'
method = 'EBCC'
# method = 'ZC'

# kv = 6
# dataname = 'rte'
dataname = 'd_sentiment'

# 获取acc_值
# 修改为正态分布的平均值
def mean_dist() -> float:
    acc_ = 0.9
    return acc_
acc_ = mean_dist()


# 获取原始数据数据长度 (如d_sentiment为1000)
# source = './dataset/' 或 source = './dataset/'
def get_length(source):
    input_file0 = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Input/'
    csvPath0 = input_file0 + 'truth.csv'
    data0 = pd.read_csv(csvPath0)
    length = len(data0)
    return length

