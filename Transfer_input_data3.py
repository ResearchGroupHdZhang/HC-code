'''
Convert the data into the form that our framework wants to input :the form of truth table.
'''


import pandas as pd
from param import param

# 主函数
if __name__ == '__main__':

    source = './dataset/'
    deta = param.deta
    low = param.low
    dataname = param.dataname
    group_num = param.group_num

    input_file1 = source + dataname + '_' + str(deta) + '_' + str(low) + '/method_initialization/'
    input_file2 = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Output/'
    input_file3 = source + dataname + '_' + str(deta) + '_' + str(low) + '/factset/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/Transfer_initial2inputdata/'

    # for method in ['EBCC', 'MV']:
    # for method in ['DS', 'MV', 'BWA', 'EBCC', 'BCC', 'ZC', 'PM']:
    for method in ['DS', 'MV', 'GLAD', 'BWA', 'EBCC', 'BCC', 'ZC', 'PM']:
        print('************:', method)
        txtPath1 = input_file1 + dataname + '_initialization_by' + method + '_' + str(deta) + '.txt'
        # txtPath1 = input_file1 + dataname + '_initialization_by'+method+'_low_0.8.txt'

        csvPath = input_file2 + dataname + '_test_gt.csv'  # 输入gt

        # 写入文件
        txtPath2 = output_file + dataname + '_Inputdata' + method + '_' + str(deta) + '.txt'
        # txtPath2 = output_file + dataname + '_Inputdata' + method + '_low_0.8.txt'


        data1 = []
        with open(txtPath1, 'r') as f:
            for line in f:
                data1.append(line.strip().split('\t'))

        data2 = pd.read_csv(csvPath, usecols=['id', 'truth'])

        factset6 = pd.read_csv(input_file3 + 'factset6.csv')  # fact为6的真值表

        factset4 = pd.read_csv(input_file3 + 'factset4.csv')

        factset2 = pd.read_csv(input_file3 + 'factset2.csv')

        factset = pd.read_csv(input_file3 + 'factset5.csv')

        factset10 = pd.read_csv(input_file3 + 'factset10.csv')

        tid = []
        gt = []

        lenght = len(data2)
        last = lenght // group_num * group_num

        # if dataname == 'd_sentiment':
        #     lenght = 1000
        #     last = 996
        # if dataname == 'rte':
        #     lenght = 690
        #     last = 690

        for i in range(len(data2)):
            tid.append(data2.iloc[i, 0])
            gt.append(data2.iloc[i, 1])

        with open(txtPath2, "w") as f:
            for j in range(lenght):
                if j % group_num == 0 and j < last:
                    f.write(str(tid[j])+'\n')

                    f10 = data1[j][1]
                    f11 = data1[j][2]
                    f20 = data1[j + 1][1]
                    f21 = data1[j + 1][2]
                    f30 = data1[j + 2][1]
                    f31 = data1[j + 2][2]
                    f40 = data1[j + 3][1]
                    f41 = data1[j + 3][2]
                    f50 = data1[j + 4][1]
                    f51 = data1[j + 4][2]
                    # f60 = data1[j + 5][1]
                    # f61 = data1[j + 5][2]
                    # f70 = data1[j + 6][1]
                    # f71 = data1[j + 6][2]
                    # f80 = data1[j + 7][1]
                    # f81 = data1[j + 7][2]
                    # f90 = data1[j + 8][1]
                    # f91 = data1[j + 8][2]
                    # f100 = data1[j + 9][1]
                    # f101 = data1[j + 9][2]

                    for s in range(len(factset)):
                        pr = 1
                        if factset.iloc[s, 0] == 1:
                            pr *= float(f11)
                        else:
                            pr *= float(f10)
                        if factset.iloc[s, 1] == 1:
                            pr *= float(f21)
                        else:
                            pr *= float(f20)
                        if factset.iloc[s, 2] == 1:
                            pr *= float(f31)
                        else:
                            pr *= float(f30)
                        if factset.iloc[s, 3] == 1:
                            pr *= float(f41)
                        else:
                            pr *= float(f40)
                        if factset.iloc[s, 4] == 1:
                            pr *= float(f51)
                        else:
                            pr *= float(f50)
                        # if factset.iloc[s, 5] == 1:
                        #     pr *= float(f61)
                        # else:
                        #     pr *= float(f60)
                        # if factset.iloc[s, 6] == 1:
                        #     pr *= float(f71)
                        # else:
                        #     pr *= float(f70)
                        # if factset.iloc[s, 7] == 1:
                        #     pr *= float(f81)
                        # else:
                        #     pr *= float(f80)
                        # if factset.iloc[s, 8] == 1:
                        #     pr *= float(f91)
                        # else:
                        #     pr *= float(f90)
                        # if factset.iloc[s, 9] == 1:
                        #     pr *= float(f101)
                        # else:
                        #     pr *= float(f100)

                        f.write(str(pr)+' ')

                # if (j + 1) % group_num == 0:
                #     f.write('\n')
                #     f.write(str(gt[j-9])+' '+str(gt[j-8])+' '+str(gt[j-7])+' '+str(gt[j-6])+' '+str(gt[j-5])+' '+str(gt[j-4])+' '+str(gt[j-3])+' '+str(gt[j-2])+' '+str(gt[j-1])+' '+str(gt[j])+'\n')

                if (j + 1) % group_num == 0:
                    f.write('\n')
                    f.write(str(str(gt[j - 4]) + ' ' + str(gt[j - 3]) + ' ' + str(gt[j - 2]) + ' ' + str(gt[j - 1]) + ' ' + str(gt[j]) + '\n'))

                if j == last:
                    # 最后剩4个
                    if lenght-last == 4:
                        f.write(str(tid[j]) + '\n')

                        f10 = data1[j][1]
                        f11 = data1[j][2]
                        f20 = data1[j + 1][1]
                        f21 = data1[j + 1][2]
                        f30 = data1[j + 2][1]
                        f31 = data1[j + 2][2]
                        f40 = data1[j + 3][1]
                        f41 = data1[j + 3][2]

                        for s in range(len(factset4)):
                            # print(len(factset4))
                            pr = 1
                            if factset.iloc[s, 0] == 1:
                                pr *= float(f11)
                            else:
                                pr *= float(f10)
                            if factset.iloc[s, 1] == 1:
                                pr *= float(f21)
                            else:
                                pr *= float(f20)
                            if factset.iloc[s, 2] == 1:
                                pr *= float(f31)
                            else:
                                pr *= float(f30)
                            if factset.iloc[s, 3] == 1:
                                pr *= float(f41)
                            else:
                                pr *= float(f40)

                            f.write(str(pr) + ' ')
                    # 最后剩两个
                    if lenght-last == 2:
                        f.write(str(tid[j]) + '\n')

                        f10 = data1[j][1]
                        f11 = data1[j][2]
                        f20 = data1[j + 1][1]
                        f21 = data1[j + 1][2]

                        for s in range(len(factset2)):
                            # print(len(factset2))
                            pr = 1
                            if factset.iloc[s, 0] == 1:
                                pr *= float(f11)
                            else:
                                pr *= float(f10)
                            if factset.iloc[s, 1] == 1:
                                pr *= float(f21)
                            else:
                                pr *= float(f20)

                            f.write(str(pr) + ' ')

                if (j + 1) == lenght:
                    if lenght - last == 4:
                        f.write('\n')
                        f.write(str(gt[j - 3]) + ' ' + str(gt[j - 2]) + ' ' + str(gt[j - 1]) + ' ' + str(gt[j]) + '\n')
                    if lenght - last == 2:
                        f.write('\n')
                        f.write(str(gt[j - 1]) + ' ' + str(gt[j]) + '\n')
