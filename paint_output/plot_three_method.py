"""
plot txt files from ./dataset/main_Output/three_method
"""
import sys
sys.path.append('/home/hwx/HC_code/')

import numpy as np
import matplotlib.pylab as plt
import math
from param import param


def open_data(str):
    with open(str, "r")as f:
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

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/three_method/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/final_plot_pdf/'

    filename_out1 = "brute"
    filename_out2 = "approx"
    filename_out3 = "random"

    # method = 'DS'
    # method = 'MV'
    # method = 'GLAD'
    # method = 'BWA'
    # method = 'PM'
    # method = 'EBCC'
    # method = 'BCC'
    # method = 'ZC'
    method = param.method

    for k in range(1, 4):
        x_list, y_list = open_data(input_file + dataname + "_0.9acc+"+ filename_out1 + "(k=" + str(k) + ")" + method + "_compare.txt")
        x_list1, y_list1 = open_data(input_file + dataname + "_0.9acc+"+ filename_out2 + "(k=" + str(k) + ")" + method + "_compare.txt")
        x_list2, y_list2 = open_data(input_file + dataname + "_0.9acc+"+ filename_out3 + "(k=" + str(k) + ")" + method + "_compare.txt")

        plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

        plt.plot(x_list, y_list, "r->", linewidth=1,label='OPT       K=' + str(k))
        plt.plot(x_list1, y_list1, "y-x", linewidth=1, label='Approx   K=' + str(k))
        plt.plot(x_list2, y_list2, "g-d", linewidth=1, label='Random K=' + str(k))

        xmax = max(x_list + x_list1)
        ymax = math.ceil(max(y_list + y_list1)) + 50
        ymin = math.floor(min(y_list + y_list1)) - 50

        # 设置坐标轴范围
        plt.xlim((0, xmax))
        plt.ylim((-300, -100))
        # 设置坐标轴名称
        plt.xlabel('Budgets',fontsize=26, fontproperties='Times New Roman', weight='bold')
        plt.ylabel('Quality',fontsize=26, fontproperties='Times New Roman', weight='bold')
        # 设置坐标轴刻度
        my_x_ticks = np.arange(0, xmax+1, 200)
        my_y_ticks = np.arange(-300, -50, 50)
        plt.xticks(my_x_ticks, size=18)
        plt.yticks(my_y_ticks, size=18)
        plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
        plt.legend(loc='lower right',prop={'family' : 'Arial', 'size': 13})
        plt.savefig(output_file + dataname + method + '_diff_method_k=' + str(k) +'.pdf', dpi=800, bbox_inches='tight')
        plt.show()
