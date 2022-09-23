"""
plot txt files from ./dataset/main_Output/outputtxt/diff_k
"""
import sys
sys.path.append('/home/hwx/HC_code/')

import numpy as np
import matplotlib.pylab as plt
import math
from param import param


def open_data(str):
    with open(str, 'r')as f:
        lines = f.readlines()
    x_list: list[int] = []
    y_list: list[float] = []
    for line in lines:
        x, y = line.split(' ')
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

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/diff_k/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/final_plot_pdf/'

    # kv=6
    for kv in range(1, group_num+2):
        x_list, y_list = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=1.txt')
        x_list1, y_list1 = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=2.txt')
        x_list2, y_list2 = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=3.txt')
        x_list3, y_list3 = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=4.txt')
        x_list4, y_list4 = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=5.txt')
        # x_list5, y_list5 = open_data(input_file + dataname + '_0.9acc+approx(k=' + str(kv) + ')' + method + '_k=6.txt')
    
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
        ymax = math.ceil(max(y_list + y_list1 + y_list2 + y_list3 + y_list4)) + 2
        ymin = math.floor(min(y_list + y_list1 + y_list2 + y_list3 + y_list4)) - 2
    
        # 设置坐标轴范围
        plt.xlim((0, xmax))
        plt.ylim((-280, 0))
        # 设置坐标轴名称
        plt.xlabel('Budgets',fontsize=22,fontproperties='Times New Roman', weight='bold')
        plt.ylabel('Quality', fontsize=22,fontproperties='Times New Roman', weight='bold')
        # 设置坐标轴刻度
        my_x_ticks = np.arange(0, xmax+1, 200)
        my_y_ticks = np.arange(-280, 40, 40)
        plt.xticks(my_x_ticks, size=18)
        plt.yticks(my_y_ticks, size=18)
        plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
        plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 13})
        plt.savefig(output_file+dataname+'_quality_kv=' + str(kv) + '_'+method+'_diff_k.pdf', dpi=800)
        plt.show()
