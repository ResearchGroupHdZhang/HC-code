'''
Output the data to be compared
'''

import pandas as pd
from collections import Counter
from param import param


def insert(df, i, df_add):
    # 指定第i行插入一行数据
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=True)
    return df_new


if __name__ == '__main__':
    source = './dataset/'
    deta = param.deta
    low = param.low
    theta = param.theta ##工人划分
    dataname = param.dataname
    group_num = param.group_num
    theta = param.theta

    input_file1 = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/'
    input_file2 = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Output/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/outcomparesdata6/'

    # method = 'EBCC'
    # method = 'DS'
    method = param.method

    for kv in range(1, group_num+2):
        # for k_ in range(1, 2):
        for k_ in range(1, group_num+1):
            datafile = input_file1 + 'exworker_select/' + dataname + '_exworker_select(k='+str(kv)+')'+method+'_'+str(deta)+'(k='+str(k_)+')'+'.csv'

            data = pd.read_csv(datafile)
            se_id = []

            for i in range(len(data)):
                se_id.append(int(data.iloc[i,0]) + int(data.iloc[i,1]))
            se_id.sort()
            print(len(se_id))

            '''找出重复的fact,把标签再加一次'''
            b = dict(Counter(se_id))
            a = {}
            for key,value in b.items():
                if value > 1:
                    a[key] = value

            if dataname == 'd_sentiment':
                step = 8 #需修改

            df = pd.read_csv(input_file2 + dataname + '_' + str(low) + '_' +str(step)+'_'+str(deta)+ '_theta' + str(theta) +'.csv')  # 文件名得记得改
            # print(df)
            dfw = []
            print(a)
            lenght = len(df)
            # print(lenght)

            if dataname == 'd_sentiment':
                sid = -1
                index1 = []
                for i in range(lenght):  # fact数
                    tid = int(df.iloc[i, 0])
                    if tid not in se_id and tid != sid:
                        for index in df[(df.id == tid) & (df.acc_d >= deta)].index:
                            index1.append(index)
                            sid = tid
                # print(index1)
                df = df.drop(set(index1))

                df.reset_index(drop=True, inplace=True)
                df1 = df

                hid = -1
                index = -1
                for j in range(len(df1)):
                    h = int(df1.iloc[j, 0])
                    if h in a.keys() and h != hid:
                        # print(j)
                        hid = h
                        insert1 = df1[(df1.id == hid) & (df1.acc_d >= deta)]
                        # print(insert1)
                        for index in df1[(df1.id == hid) & (df1.acc_d >= deta)].index:
                            index = index
                        # print(insert)
                        for time in range(a[h] - 1):
                            # print(index)
                            df1 = insert(df1, index, insert1)
                df1 = df1.drop(['truth', 'acc_d'], axis=1)
                df1.to_csv(output_file + dataname + '_select_final(k=' + str(kv) + ')' + method + '_' + str(deta) + '(k=' + str(k_) + ')' + '.csv', index=False, encoding='utf-8')

        if dataname == 'rte':
                sid = -1
                index1 = []
                for i in range(lenght):  # fact数
                    tid = int(df.iloc[i, 0])
                    if tid not in se_id and tid != sid:
                        for index in df[(df.id == tid)&(df.acc_d >= deta)].index:
                            index1.append(index)
                            sid = tid
                # print(index1)
                df = df.drop(set(index1))
                # print(df)

                df.reset_index(drop=True, inplace=True)
                df1 = df

                hid = -1
                index = -1
                for j in range(len(df1)):
                    h = int(df1.iloc[j, 0])
                    if h in a.keys() and h != hid:
                        # print(j)
                        hid = h
                        insert1 = df1[(df1.id == hid)&(df1.acc_d >= deta)]
                        # print(insert1)
                        for index in df1[(df1.id == hid) & (df1.acc_d >= deta)].index:
                            index = index
                        # print(insert)
                        for time in range(a[h] - 1):
                            # print(index)
                            df1 = insert(df1, index, insert1)
                df1 = df1.drop(['truth', 'acc_d'], axis=1)
                df1.to_csv(output_file + dataname + '_select_final(k='+str(kv)+')'+method+'_'+str(deta)+'(k='+str(k_)+')'+'.csv', index=False, encoding='utf-8')

