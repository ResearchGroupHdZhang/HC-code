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

    # metric = 'accuracy'
    # metric = 'F1_score'
    metric = 'Quality'

    input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/main_Output/outputtxt/diff_init/'
    output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/final_plot_pdf/'

    # kv=6
    for k in range(1, 7):
        x_list, y_list = open_data(input_file + dataname + '_0.9acc+approx(k=6)MV_k=' + str(k) + '.txt')
        x_list1, y_list1 = open_data(input_file + dataname + '_0.9acc+approx(k=6)DS_k=' + str(k) + '.txt')
        x_list2, y_list2 = open_data(input_file + dataname + '_0.9acc+approx(k=6)ZC_k=' + str(k) + '.txt')
        x_list3, y_list3 = open_data(input_file + dataname + '_0.9acc+approx(k=6)GLAD_k=' + str(k) + '.txt')
        x_list4, y_list4 = open_data(input_file + dataname + '_0.9acc+approx(k=6)BWA_k=' + str(k) + '.txt')
        x_list5, y_list5 = open_data(input_file + dataname + '_0.9acc+approx(k=6)BCC_k=' + str(k) + '.txt')
        x_list6, y_list6 = open_data(input_file + dataname + '_0.9acc+approx(k=6)EBCC_k=' + str(k) + '.txt')
        x_list7, y_list7 = open_data(input_file + dataname + '_0.9acc+approx(k=6)PM_k=' + str(k) + '.txt')
        plt.figure(figsize=(10,6), dpi=200, facecolor='w', edgecolor='k')

        plt.plot(x_list, y_list, 'c-+', linewidth=1, label='MV_INIT')
        plt.plot(x_list1, y_list1, 'b-o', linewidth=1, label='DS_INIT')
        plt.plot(x_list2, y_list2, 'y-*', linewidth=1, label='ZC_INIT')
        plt.plot(x_list3, y_list3, 'm-x', linewidth=1, label='GLAD_INIT')
        plt.plot(x_list4, y_list4, 'g-s', linewidth=1, label='BWA_INIT')
        plt.plot(x_list5, y_list5, 'm-.', linewidth=1, label='BCC_INIT')
        plt.plot(x_list6, y_list6, 'k-d', linewidth=1, label='EBCC_INIT')
        plt.plot(x_list7, y_list7, color='0.5', marker = '>', linewidth=1, label='CRH_INIT')

        xmax = max(x_list+ x_list1+ x_list2+ x_list3+ x_list4+ x_list5+ x_list6+ x_list7)
        ymax = math.ceil(max(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5+ y_list6+ y_list7)) + 2
        ymin = math.floor(min(y_list+ y_list1+ y_list2+ y_list3+ y_list4+ y_list5+ y_list6+ y_list7)) - 2

        # 设置坐标轴范围
        plt.xlim((0, xmax))
        plt.ylim((ymin, 0))
        # 设置坐标轴名称
        plt.xlabel('Budgets_k=' + str(k), fontsize=22,fontproperties='Times New Roman', weight='bold')
        plt.ylabel(metric, fontsize=22,fontproperties='Times New Roman', weight='bold')
        # 设置坐标轴刻度
        my_x_ticks = np.arange(0, xmax+1, 400)
        my_y_ticks = np.arange(ymin, ymax, 100)
        plt.xticks(my_x_ticks, size=16)
        plt.yticks(my_y_ticks, size=16)
        plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
        plt.legend(loc='lower right',prop={'family' : 'Arial', 'size': 13})
        plt.savefig(output_file + dataname +'_'+ metric + '_compare_INIT_k=' + str(k) + '.pdf', dpi=800)
        plt.show()
