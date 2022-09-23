"""
plot txt files from ./dataset/cal_f1/diff_k
"""
import sys
sys.path.append('/home/hwx/HC_code/')

import numpy as np
import matplotlib.pylab as plt
import math
from param import param
from matplotlib.ticker import FuncFormatter


def open_data(str):
    with open(str, 'r')as f:
        lines = f.readlines()[1:]
    x_list: list[int] = []
    y_list: list[float] = []
    for line in lines:
        x, y = line.split(',')
        x_list.append(int(x))
        y_list.append(float(y))
    return x_list, y_list


if __name__ == '__main__':
    source = './dataset/'

    deta = param.deta
    low = param.low
    dataname = param.dataname
    group_num = param.group_num

    # method = 'DS'
    # method = 'MV'
    # method = 'GLAD'
    # method = 'BWA'
    # method = 'PM'
    # method = 'EBCC'
    # method = 'BCC'
    # method = 'ZC'
    method = param.method

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/cal_f1/diff_k/accuracy/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/final_plot_pdf/'

    x_list, y_list = open_data(input_file + dataname + '(k=' + str(group_num+1) + ')_accuracy_HC_' + method + '_k=1_' + str(deta) + '.txt')
    x_list1, y_list1 = open_data(input_file + dataname + '(k=' + str(group_num+1) + ')_accuracy_HC_' + method + '_k=2_' + str(deta) + '.txt')
    x_list2, y_list2 = open_data(input_file + dataname + '(k=' + str(group_num+1) + ')_accuracy_HC_' + method + '_k=3_' + str(deta) + '.txt')
    x_list3, y_list3 = open_data(input_file + dataname + '(k=' + str(group_num+1) + ')_accuracy_HC_' + method + '_k=4_' + str(deta) + '.txt')
    x_list4, y_list4 = open_data(input_file + dataname + '(k=' + str(group_num+1) + ')_accuracy_HC_' + method + '_k=5_' + str(deta) + '.txt')
    # x_list5, y_list5 = open_data(input_file + dataname + '(k=' + str(group_num) + ')_accuracy_HC_' + method + '_k=6_' + str(deta) + '.txt')
    # d_sentiment(k=1)_accuracy_HC_EBCC.txt

    plt.figure(figsize=(10,6), dpi=200, facecolor='w', edgecolor='k')

    plt.plot(x_list, y_list, 'c-+', linewidth=1, label='K=1')
    plt.plot(x_list1, y_list1, 'r-o', linewidth=1, label='K=2')
    plt.plot(x_list2, y_list2, 'y-*', linewidth=1, label='K=3')
    plt.plot(x_list3, y_list3, 'm-x', linewidth=1, label='K=4')
    plt.plot(x_list4, y_list4, 'g-s', linewidth=1, label='K=5')
    # plt.plot(x_list5, y_list5, 'b-x', linewidth=1, label='K=6')

    # 找到绘图边缘值 group_num = 6
    # xmax = max(x_list+ x_list1+ x_list2+ x_list3+ x_list4+ x_list5)
    # ymax = math.ceil(max(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5)) + 2
    # ymin = math.floor(min(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5)) - 2

    # group_num = 5
    xmax = max(x_list + x_list1 + x_list2 + x_list3 + x_list4)
    ymax = math.ceil(max(y_list + y_list1 + y_list2 + y_list3 + y_list4))
    ymin = math.floor(min(y_list + y_list1 + y_list2 + y_list3 + y_list4))

    # 设置坐标轴范围
    plt.xlim((0, xmax))
    plt.ylim((ymin, ymax))

    # 纵轴显示百分比
    def to_percent(temp, position):
        return '%1.0f' % temp + '%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    # 设置坐标轴名称
    plt.xlabel('Budgets',fontsize=26)
    plt.ylabel('Accuracy',fontsize=26)
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, xmax+1, 200)
    my_y_ticks = np.arange(ymin, ymax+2, 2)
    plt.xticks(my_x_ticks, size=18)
    plt.yticks(my_y_ticks, size=18)
    plt.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 13})
    plt.savefig(output_file+dataname+'_accuracy'+method+'_diff_k.pdf', dpi=800)
    plt.show()
