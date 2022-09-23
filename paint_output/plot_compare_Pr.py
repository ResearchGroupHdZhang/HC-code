import numpy as np
import matplotlib.pylab as plt
import math
from param import param


def open_data(str):
    with open(str, 'r')as f:
        # next(f)
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
    # dataname = 'd_sentiment'

    metric = 'Quality'

    # if dataname == 'd_sentiment':
    input_file1 = source + dataname + '_0.85_0.9theta0.9' + '/main_Output/outputtxt/three_method/'
    input_file2 = source + dataname + '_0.85_0.9theta0.85' + '/main_Output/outputtxt/three_method/'
    input_file3 = source + dataname + '_0.85_0.9theta0.8' + '/main_Output/outputtxt/three_method/'
    output_file = source

    # if dataname == 'rte':
    #     input_file1 = source + dataname + '_0.8_0.8' + '/main_Output/outputtxt/three_method/'
    #     input_file2 = source + dataname + '_0.9_0.9' + '/main_Output/outputtxt/three_method/'
    #     output_file = source

    for k in range(1, ):
        x_list, y_list = open_data(input_file1 + dataname + '_0.9acc+approx(k='+str(k)+')EBCC_compare.txt')
        x_list1, y_list1 = open_data(input_file2 + dataname + '_0.9acc+approx(k=' + str(k) + ')EBCC_compare.txt')
        x_list2, y_list2 = open_data(input_file3 + dataname + '_0.9acc+approx(k=' + str(k) + ')EBCC_compare.txt')
        # x_list3, y_list3 = open_data('./output/' + dataname + '_0.9acc+approx(k=' + str(k) + ')EBCC_low_0.8_0.9_k=6.txt')

        plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

        # if dataname == 'd_sentiment':
        plt.plot(x_list, y_list, 'c-+', linewidth=1, label='θ = 0.8')
        plt.plot(x_list1, y_list1, 'm-o', linewidth=1, label='θ = 0.9')
        # if dataname == 'rte':
        #     plt.plot(x_list, y_list, 'c-+', linewidth=1, label='θ = 0.75')
        #     plt.plot(x_list1, y_list1, 'm-o', linewidth=1, label='θ = 0.8')

        xmax = max(x_list + x_list1)
        ymax = math.ceil(max(y_list + y_list1)) + 100
        ymin = math.floor(min(y_list + y_list1)) - 50

        # 设置坐标轴范围
        plt.xlim((0, xmax))
        plt.ylim((ymin, ymax))
        # 设置坐标轴名称
        plt.xlabel('Budgets_k=' + str(k),fontsize=26,fontproperties='Times New Roman', weight='bold')
        plt.ylabel(metric, fontsize=26,fontproperties='Times New Roman', weight='bold')
        # 设置坐标轴刻度
        my_x_ticks = np.arange(0, xmax+1, 400)
        my_y_ticks = np.arange(ymin, ymax, 20)
        plt.xticks(my_x_ticks, size=18)
        plt.yticks(my_y_ticks, size=18)
        plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
        plt.legend(loc='lower right',prop={'family' : 'Arial', 'size': 18})
        plt.savefig(output_file + dataname +'_'+ metric + '_compare_pr_k='+str(k)+'.pdf', dpi=800, bbox_inches='tight')
        plt.show()
