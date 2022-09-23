'''
Processing data
Output initial training data:d_sentiment_test5.csv  and probability of expert workers:d_sentiment_worker_acc.txt
The data sets that will be used later are all stored in 'datasets'.
'''

from numpy import *
import pandas as pd
from param import param

import sys 
sys.path.append('/home/hwx/HC_code/')

source = './dataset/'

deta = param.deta
low = param.low
dataname = param.dataname
group_num = param.group_num
# **zhenh** theta
theta = param.theta

input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Input/'
output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Output/'

# ----------------------------------------------------------
datafile = input_file + 'label.csv'
datatruth = input_file + 'truth.csv'

answer = pd.read_csv(datafile)
answer.columns = ['id', 'worker', 'answer']
truth = pd.read_csv(datatruth)

full = pd.merge(answer, truth, left_on='id', right_on='item', how='left')

full = full.drop(['item'], axis=1)
full.to_csv(output_file + 'full.csv', index=False, encoding="utf-8")


def compute_acc(df1):
    return round(len(df1[df1['answer'] == df1['truth']])/len(df1),2)

acc_data = pd.DataFrame(columns=['acc_d'])
acc_data['acc_d'] = full.groupby('worker').apply(compute_acc)

acc_data.to_csv(output_file + 'worker_acc.csv', encoding="utf-8")

full_acc = pd.merge(full, acc_data, left_on='worker', right_on='worker', how='left')
full_acc.sort_values(['id', 'acc_d'], axis=0, ascending=True, inplace=True)
full_acc.to_csv(output_file + 'full_acc.csv', index=False, encoding="utf-8")

df = pd.read_csv(output_file + 'full_acc.csv')
df.head(2)

df_low = pd.read_csv(input_file + dataname + '_low_' + str(deta) + '.csv')

# df4:仅包含fact_id, gt的集合
df4 = df_low.drop_duplicates(subset=['id', 'truth'], keep='first', inplace=False)
df4 = df4.drop(['worker', 'answer', 'acc_d'], axis=1)
df4.to_csv(output_file + dataname + '_test_gt.csv', index=False, encoding="utf-8")

if dataname == 'd_sentiment':
    df_lo = df_low.groupby('id').tail(6)  # 筛选后6个，概率都小于deta
    df_lo = df_lo.groupby('id').head(6)  # 筛选后6个，概率都小于deta
    step = 8  # hwx: step是【CP+CE数量】，工人总数

# hwx: 这边可以按照param修改了theta的情况改
if low == 0.9:
    df_hi = pd.read_csv(input_file + dataname + '_high_0.9.csv')
if low == 0.8:
    df_hi = pd.read_csv(input_file + dataname + '_low_0.8_high_0.9.csv')
if low == 0.85:
    df_hi = pd.read_csv(input_file + dataname + '_high_0.85.csv')

df_hi_l = pd.read_csv(input_file + dataname + '_high_0.85.csv')
df_hi2 = df_hi_l.groupby('id').head(1)
# df_hi2 = df_hi2.groupby('id').tail(1)

# **zhenh** 不用什么low high了， 这里deta只是一个挑选工人的手段
df_hi1 = df_hi.groupby('id').head(1)
# df_hi1 = df_hi1.groupby('id').tail(2)

df_hi1 = pd.concat([df_hi2, df_hi1])

df_lo_hi = pd.concat([df_lo, df_hi1])
df_lo_hi.sort_values(['id', 'acc_d'], axis=0, ascending=True, inplace=True)
# **zhenh**
# df_lo_hi:工人集合(step个) 修改名字: 包括 outputcomparedata6.py 文件
df_lo_hi.to_csv(output_file + dataname + '_'+str(low)+'_'+str(step)+'_'+str(deta) + '_theta' + str(theta)+'.csv', index=False, encoding="utf-8")

# **zhenh** theta筛选 df_hi1:高于theta的工人 '_worker_ex'好像没用到
df_hi1 = df_lo_hi[df_lo_hi['acc_d'] >= theta]
df_hi1.to_csv(output_file + dataname + '_worker_ex' + str(low) + '_theta' + str(theta) + '.csv', index=False, encoding="utf-8")
# df_lo:低于theta的工人集合 '_6.csv' 好像没用到
df_lo = df_lo_hi[df_lo_hi['acc_d'] < theta]
df_lo.to_csv(output_file + dataname + '_low_' + str(deta) + '_theta' + str(theta) + '_6.csv', index=False, encoding="utf-8")
# df2:仅包含fact_id, worker_id, answer的集合 修改名字: 包括 method_*2.py 文件
df2 = df_lo.drop(['truth', 'acc_d'], axis=1)
df2.to_csv(output_file + dataname + '_test6_' + str(deta) + '_theta' + str(theta) + '.csv', index=False, encoding="utf-8")

#看工人概率的改变
pr_sum1 = 0
pr_sum2 = 0
pr_sum = 0
for i in range(0,len(df_lo_hi),step):
    for j in range(0,step-2):
        pr_sum += float(df_lo_hi.iloc[i + j, 4])

    pr_sum1 += float(df_lo_hi.iloc[i + step-2, 4])
    pr_sum2 += float(df_lo_hi.iloc[i + step-1, 4])

print((pr_sum/(len(df_lo_hi)/step*(step-2))))
print((pr_sum1/(len(df_lo_hi)/step)))
print((pr_sum2/(len(df_lo_hi)/step)))
print(((pr_sum1+pr_sum2)/(len(df_lo_hi)/step*2)))


df5 = df_hi1
# 修改名字: 包含processor*.py, main*.py文件
fact_num = param.fact_num
with open(output_file + dataname + '_worker_acc' + str(deta) + '_' + str(low) + '_theta' + str(theta) + '.txt', "w") as f:
    # d_senti: group_num = 5 ; len(df5)
    for i in range(0, fact_num, group_num):
        df_id = df5[df5['id'] == i]
        expert_num = 0
        if len(df_id) != 0:
            f.write(str(df_id.iloc[0, 0]))
            f.write('\n')
            expert_num = len(df_id)
            for j in range(expert_num):
                f.write(str(df_id.iloc[j, 4]))
                f.write(' ')
            f.write('\n')
        if expert_num != 0:
            expert_answer = []
            for j in range(expert_num):
                expert_answer.append([])
            for j in range(group_num):
                df_id = df5[df5['id'] == (i+j)]
                for expert_index in range(expert_num):
                    expert_answer[expert_index].append(df_id.iloc[expert_index, 2])
            for expert_index in range(expert_num):
                for answer in range(len(expert_answer[expert_index])):
                    f.write(str(expert_answer[expert_index][answer]))
                    f.write(' ')
                f.write('\n')
