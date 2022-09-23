import sys
sys.path.append('/home/hwx/HC_code/')

import numpy as np
import matplotlib.pylab as plt
import math
from param import param
from matplotlib.ticker import FuncFormatter


def open_data(str):
    with open(str, 'r')as f:
        # next(f)
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

    metric = 'accuracy'
    # metric = 'F1_score'
    # metric = 'Quality'

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/cal_f1/diff_init/accuracy/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/final_plot_pdf/'

    for k in range(1, group_num+1):
        x_list, y_list = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_MV_EBCC_' + str(deta) +'.txt')
        x_list1, y_list1 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_DS_EBCC_' + str(deta) +'.txt')
        x_list2, y_list2 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_ZC_EBCC_' + str(deta) +'.txt')
        x_list3, y_list3 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_GLAD_EBCC_' + str(deta) +'.txt')
        x_list4, y_list4 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_BWA_EBCC_' + str(deta) +'.txt')
        x_list5, y_list5 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_BCC_EBCC_' + str(deta) +'.txt')
        x_list6, y_list6 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_EBCC_EBCC_' + str(deta) +'.txt')
        x_list7, y_list7 = open_data(input_file + dataname + '(k=' + str(k) + ')_accuracy_PM_EBCC_' + str(deta) +'.txt')
        plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')
        print(len(x_list))

        plt.plot(x_list, y_list, 'c-+', linewidth=1, label='MV')
        plt.plot(x_list1, y_list1, 'b-o', linewidth=1, label='DS')
        plt.plot(x_list2, y_list2, 'y-*', linewidth=1, label='ZC')
        plt.plot(x_list3, y_list3, 'm-x', linewidth=1, label='GLAD')
        plt.plot(x_list4, y_list4, 'g-s', linewidth=1, label='BWA')
        plt.plot(x_list5, y_list5, 'm-.', linewidth=1, label='BCC')
        plt.plot(x_list6, y_list6, 'k-d', linewidth=1, label='EBCC')
        plt.plot(x_list7, y_list7, color='0.5', marker = '>', linewidth=1, label='CRH')

        xmax = max(x_list+ x_list1+ x_list2+ x_list3+ x_list4+ x_list5+ x_list6+ x_list7)
        ymax = math.ceil(max(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5+ y_list6+ y_list7)) + 2
        ymin = math.floor(min(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5+ y_list6+ y_list7)) - 2

        # 设置坐标轴范围
        plt.xlim((0, xmax))
        plt.ylim((ymin, ymax))

        # 纵轴显示百分比
        def to_percent(temp, position):
            return '%1.0f' % temp + '%'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        # plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

        # 设置坐标轴名称
        plt.xlabel('Budgets',fontsize=22,fontproperties='Times New Roman', weight='bold')
        plt.ylabel(metric, fontsize=22,fontproperties='Times New Roman', weight='bold')
        # 设置坐标轴刻度
        my_x_ticks = np.arange(0, xmax + 1, 300)
        my_y_ticks = np.arange(ymin, ymax+2, 2)
        plt.xticks(my_x_ticks, size=16)
        plt.yticks(my_y_ticks, size=16)
        plt.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
        plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 13})
        plt.savefig(output_file + dataname + '_' + metric + '_compare_INIT_k=' + str(k) + '_' + str(deta) + '.pdf', dpi=800)
        plt.show()
