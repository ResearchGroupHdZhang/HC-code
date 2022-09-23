from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
from param import param

source = './dataset/'

deta = param.deta
low = param.low
dataname = param.dataname
length = param.get_length(source)
group_num = param.group_num

input_file0 = source + dataname + '_' + str(deta) + '_' + str(low) + '/Preliminary_processing_data1_Output/'
input_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/output_HC_label5/'
output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/cal_f1/diff_k/'

# method = 'DS'
# method = 'EBCC'
method = param.method

truePath = input_file0 + dataname +'_test_gt.csv'

for k in range (1, group_num+1):
    budget = []

    f1_HC = []

    acc_HC = []
    for kv in range(1,group_num+2):
        print('***************************', kv, k)
        if dataname == 'rte':
            budget.append(length//group_num * kv)
        if dataname == 'd_sentiment':
            # if kv == 1:
            #     budget.append((length//6+1) * kv)
            # else:
            budget.append((length//group_num+1) * kv)

        # 对比不同的k
        predPathHC = input_file + dataname + '_result_HC(k=' + str(kv) + ')' + method + '_k=' + str(k) + '_' + str(deta) + '.csv'

        y_true = pd.read_csv(truePath, usecols=['truth'])
        y_predHC = pd.read_csv(predPathHC, usecols=['label'])

        macro_f1_HC = f1_score(y_true, y_predHC, average='macro')

        f1_HC.append(round(macro_f1_HC*100,2))

        print(dataname + '_macro_f1_HC:\t'+str(round(macro_f1_HC*100,2)))

        accuracy_HC = accuracy_score(y_true, y_predHC)

        acc_HC.append(round(accuracy_HC*100,2))

        print(dataname + '_accuracy_HC:\t'+str(round(accuracy_HC*100,2)))

    dataHC = pd.DataFrame({'budget': budget, 'f1_score': f1_HC})
    dataHC.to_csv(output_file + 'f1_score/' +dataname+'(k=' + str(kv) + ')_F1_score_HC_k=' + str(k) + '_' + str(deta) +'.txt', index=False, sep=',')

    dataHC1 = pd.DataFrame({'budget': budget, 'accuracy_score': acc_HC})
    dataHC1.to_csv(output_file + 'accuracy/' +dataname+'(k=' + str(kv) + ')_accuracy_HC_' + method + '_k=' + str(k) + '_'  + str(deta) +'.txt', index=False, sep=',')
