"""
csv files from dataset/main_Output/outputcsv
to
txt files dataset/main_Output/outputtxt
"""
import pandas as pd
from param import param
import math

# filename_out1 = 'brute'
# filename_out2 = 'approx'
filename_out = ['brute', 'approx', 'random']
# filename_out = ['approx']

source = './dataset/'

deta = param.deta
low = param.low
dataname = param.dataname
acc = param.acc_
group_num = param.group_num  # 分组数量
length = param.get_length(source)  # 数据长度

input_file1 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputcsv/three_method/'
input_file2 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputcsv/diff_k/'
input_file3 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputcsv/diff_init/'
output_file1 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/three_method/'
output_file2 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/diff_k/'
output_file3 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/diff_init/'

# method = 'DS'
# method = 'MV'
# method = 'GLAD'
# method = 'BWA'
# method = 'PM'
# method = 'EBCC'
# method = 'BCC'
# method = 'ZC'
method = param.method

# 不同method比较
# for fn_out in filename_out:
#     for k in range(1, 4):
#         filename = input_file1 + dataname + '_' + str(acc) + 'acc+' + fn_out + '(k=' + str(k) + ')' + method + '_compare.csv'

#         x = pd.read_csv(filename)

#         with open(output_file1 + dataname + '_' + str(acc) + 'acc+' + fn_out + '(k=' + str(k) + ')' + method + '_compare.txt', 'w') as f:
#             id = 0
#             step = 20 // k
#             for i in range(0, len(x), step):
#                 f.write(str(x.iloc[i, 0] * k))
#                 # f.write(str(id))
#                 f.write(' ')
#                 # id += step * 3 * 2 * 2
#                 f.write(str(-x.iloc[i, 1]))
#                 f.write('\n')
#             f.write(str(math.ceil(length / group_num) * group_num) + ' ' + str(-x.iloc[len(x) - 1, 1]))

# 不同k比较 diff_k
for kv in range(1, group_num+2):
    for k in range(1, group_num+1):
        filename = input_file2 + dataname + '_' + str(acc) + 'acc+approx(k=' + str(kv) + ')' + method + '_k=' + str(k) + '.csv'

        x = pd.read_csv(filename)
        # print(len(x))

        with open(output_file2 + dataname + '_' + str(acc) + 'acc+approx(k=' + str(kv) + ')' + method + '_k=' + str(k) + '.txt', 'w') as f:
            id = 0
            step = 60 // k
            for i in range(0, len(x), step):
            # for i in range(0, len(x)):  
                # f.write(str(i))
                f.write(str(x.iloc[i, 0] * k))
                f.write(' ')
                # id += step * k
                f.write(str(-x.iloc[i, 1]))
                f.write('\n')
            f.write(str(math.ceil(length / group_num) * kv) + ' ' + str(-x.iloc[len(x) - 1, 1]))

# 不同初始化比较
# for kv in range(1, group_num+2):
#     # for method in ['EBCC']:
#     for method in ['DS','MV','GLAD','BWA','EBCC','BCC','ZC','PM']:
#         for k in range(1, group_num+1):
#             filename = input_file3 + dataname + '_' + str(acc) + 'acc+approx(k=' + str(kv) + ')' + method + '_k=' + str(k) + '.csv'

#             x = pd.read_csv(filename)

#             with open(output_file3 + dataname + '_' + str(acc) + 'acc+approx(k=' + str(kv) + ')' + method + '_k=' + str(k) + '.txt', 'w') as f:
#                 id = 0
#                 step = 60 // k
#                 for i in range(0, len(x), step):
#                     f.write(str(id))
#                     f.write(' ')
#                     id += step * 2 * k
#                     f.write(str(-x.iloc[i, 1]))
#                     f.write('\n')
#                 #
#                 f.write(str(math.ceil(length / group_num) * group_num*2) + ' ' + str(-x.iloc[len(x) - 1, 1]))
