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
output_file = source + dataname + '_' + str(deta) + '_' + str(low) + '/cal_f1/diff_init/'

# method = 'DS'
# method = 'EBCC'
method = param.method

truePath = input_file0 + dataname +'_test_gt.csv'

for k in range(1, group_num+1):
    budget = []

    f1_MV = []
    f1_DS = []
    f1_ZC = []
    f1_GLAD = []
    f1_PM= []
    f1_BWA = []
    f1_BCC = []
    f1_EBCC = []
    f1_HC = []

    acc_MV = []
    acc_DS = []
    acc_ZC = []
    acc_GLAD = []
    acc_PM = []
    acc_BWA = []
    acc_BCC = []
    acc_EBCC = []
    acc_HC = []

    for kv in range(1, group_num+2):
        print('***************************', kv, k)
        if dataname == 'rte':
            budget.append(length//group_num * kv)
        if dataname == 'd_sentiment':
            # if kv == 1:
            #     budget.append((length//6+1) * kv)
            # else:
            budget.append((length//group_num+1) * kv)

        # # #i对比不同初始化方法
        predPathDS = input_file + dataname + '_result_HC(k=' + str(kv) + ')DS_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathZC = input_file + dataname + '_result_HC(k=' + str(kv) + ')ZC_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathMV = input_file + dataname + '_result_HC(k=' + str(kv) + ')MV_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathGLAD = input_file + dataname + '_result_HC(k='+str(kv)+')GLAD_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathPM = input_file + dataname + '_result_HC(k=' + str(kv) + ')PM_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathBWA = input_file + dataname + '_result_HC(k='+str(kv)+')BWA_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathBCC = input_file + dataname + '_result_HC(k='+str(kv)+')BCC_k=' + str(k) + '_' + str(deta)+'.csv'
        predPathEBCC = input_file + dataname + '_result_HC(k='+str(kv)+')EBCC_k=' + str(k) + '_' + str(deta)+'.csv'

        y_true = pd.read_csv(truePath, usecols=['truth'])
        y_predMV = pd.read_csv(predPathMV, usecols=['label'])
        y_predDS = pd.read_csv(predPathDS, usecols=['label'])
        y_predZC = pd.read_csv(predPathZC, usecols=['label'])
        y_predGLAD = pd.read_csv(predPathGLAD, usecols=['label'])
        y_predPM = pd.read_csv(predPathPM, usecols=['label'])
        y_predBWA = pd.read_csv(predPathBWA, usecols=['label'])
        y_predBCC = pd.read_csv(predPathBCC, usecols=['label'])
        y_predEBCC = pd.read_csv(predPathEBCC, usecols=['label'])
        # y_predHC = pd.read_csv(predPathHC, usecols=['label'])

        macro_f1_DS = f1_score(y_true, y_predDS, average='macro')
        macro_f1_ZC = f1_score(y_true, y_predZC, average='macro')
        macro_f1_MV = f1_score(y_true, y_predMV, average='macro')
        macro_f1_GLAD = f1_score(y_true, y_predGLAD, average='macro')
        macro_f1_PM = f1_score(y_true, y_predPM, average='macro')
        macro_f1_BWA = f1_score(y_true, y_predBWA, average='macro')
        macro_f1_BCC = f1_score(y_true, y_predBCC, average='macro')
        macro_f1_EBCC = f1_score(y_true, y_predEBCC, average='macro')
        # macro_f1_HC = f1_score(y_true, y_predHC, average='macro')

        f1_MV.append(round(macro_f1_MV*100,2))
        f1_DS.append(round(macro_f1_DS*100,2))
        f1_ZC.append(round(macro_f1_ZC*100,2))
        f1_GLAD.append(round(macro_f1_GLAD*100,2))
        f1_PM.append(round(macro_f1_PM * 100, 2))
        f1_BWA.append(round(macro_f1_BWA*100,2))
        f1_BCC.append(round(macro_f1_BCC*100,2))
        f1_EBCC.append(round(macro_f1_EBCC*100,2))
        # f1_HC.append(round(macro_f1_HC*100,2))

        print(dataname + '_macro_f1_MV:\t'+str(round(macro_f1_MV*100,2)))
        print(dataname + '_macro_f1_DS:\t'+str(round(macro_f1_DS*100,2)))
        print(dataname + '_macro_f1_ZC:\t'+str(round(macro_f1_ZC*100,2)))
        print(dataname + '_macro_f1_GLAD\t:'+str(round(macro_f1_GLAD*100,2)))
        print(dataname + '_macro_f1_PM:\t' + str(round(macro_f1_PM * 100, 2)))
        print(dataname + '_macro_f1_BWA:\t'+str(round(macro_f1_BWA*100,2)))
        print(dataname + '_macro_f1_BCC:\t'+str(round(macro_f1_BCC*100,2)))
        print(dataname + '_macro_f1_EBCC:\t'+str(round(macro_f1_EBCC*100,2)))
        # print(dataname + '_macro_f1_HC:\t'+str(round(macro_f1_HC*100,2)))

        accuracy_DS = accuracy_score(y_true, y_predDS)
        accuracy_ZC = accuracy_score(y_true, y_predZC)
        accuracy_MV = accuracy_score(y_true, y_predMV)
        accuracy_GLAD = accuracy_score(y_true, y_predGLAD)
        accuracy_PM = accuracy_score(y_true, y_predPM)
        accuracy_BWA = accuracy_score(y_true, y_predBWA)
        accuracy_BCC = accuracy_score(y_true, y_predBCC)
        accuracy_EBCC = accuracy_score(y_true, y_predEBCC)
        # accuracy_HC = accuracy_score(y_true, y_predHC)

        acc_MV.append(round(accuracy_MV*100,2))
        acc_DS.append(round(accuracy_DS*100,2))
        acc_ZC.append(round(accuracy_ZC*100,2))
        acc_GLAD.append(round(accuracy_GLAD*100,2))
        acc_PM.append(round(accuracy_PM * 100, 2))
        acc_BWA.append(round(accuracy_BWA*100,2))
        acc_BCC.append(round(accuracy_BCC*100,2))
        acc_EBCC.append(round(accuracy_EBCC*100,2))
        # acc_HC.append(round(accuracy_HC*100,2))

        print(dataname + '_accuracy_MV:\t'+str(round(accuracy_MV*100,2)))
        print(dataname + '_accuracy_DS:\t'+str(round(accuracy_DS*100,2)))
        print(dataname + '_accuracy_ZC:\t'+str(round(accuracy_ZC*100,2)))
        print(dataname + '_accuracy_GLAD:\t'+str(round(accuracy_GLAD*100,2)))
        print(dataname + '_accuracy_PM:\t' + str(round(accuracy_PM * 100, 2)))
        print(dataname + '_accuracy_BCC:\t'+str(round(accuracy_BCC*100,2)))
        print(dataname + '_accuracy_EBCC:\t'+str(round(accuracy_EBCC*100,2)))
        print(dataname + '_accuracy_BWA:\t'+str(round(accuracy_BWA*100,2)))
        # print(dataname + '_accuracy_HC:\t'+str(round(accuracy_HC*100,2)))

    dataDS = pd.DataFrame({'budget': budget, 'f1_score': f1_DS})
    dataDS.to_csv(output_file + 'f1_score/' + dataname+'(k=' + str(k) + ')_F1_score_DS_' + str(deta) +'.txt', index=False, sep=',')

    dataZC = pd.DataFrame({'budget': budget, 'f1_score': f1_ZC})
    dataZC.to_csv(output_file + 'f1_score/' + dataname+'(k=' + str(k) + ')_F1_score_ZC_' + str(deta) +'.txt', index=False, sep=',')

    dataMV = pd.DataFrame({'budget': budget, 'f1_score': f1_MV})
    dataMV.to_csv(output_file + 'f1_score/' +dataname+'(k=' + str(k) + ')_F1_score_MV_' + str(deta) +'.txt', index=False, sep=',')

    dataGLAD = pd.DataFrame({'budget': budget, 'f1_score': f1_GLAD})
    dataGLAD.to_csv(output_file + 'f1_score/' + dataname+'(k=' + str(k) + ')_F1_score_GLAD_' + str(deta) +'.txt', index=False, sep=',')

    dataPM = pd.DataFrame({'budget': budget, 'f1_score': f1_PM})
    dataPM.to_csv(output_file + 'f1_score/' + dataname+'(k=' + str(k) + ')_F1_score_PM_' + str(deta) +'.txt', index=False, sep=',')

    dataBCC = pd.DataFrame({'budget': budget, 'f1_score': f1_BCC})
    dataBCC.to_csv(output_file + 'f1_score/' + dataname+'(k=' + str(k) + ')_F1_score_BCC_' + str(deta) +'.txt', index=False, sep=',')

    dataBWA = pd.DataFrame({'budget': budget, 'f1_score': f1_BWA})
    dataBWA.to_csv(output_file + 'f1_score/' +dataname+'(k=' + str(k) + ')_F1_score_BWA_' + str(deta) +'.txt', index=False, sep=',')

    dataEBCC = pd.DataFrame({'budget': budget, 'f1_score': f1_EBCC})
    dataEBCC.to_csv(output_file + 'f1_score/' +dataname+'(k=' + str(k) + ')_F1_score_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    # dataHC = pd.DataFrame({'budget': budget, 'f1_score': f1_HC})
    # dataHC.to_csv(output_file + 'f1_score/' +dataname+'(k=' + str(k) + ')_F1_score_HC_' + str(deta) +'.txt', index=False, sep=',')


    dataDS1 = pd.DataFrame({'budget': budget, 'accuracy': acc_DS})
    dataDS1.to_csv(output_file + 'accuracy/' + dataname+'(k=' + str(k) + ')_accuracy_DS_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataZC1 = pd.DataFrame({'budget': budget, 'accuracy': acc_ZC})
    dataZC1.to_csv(output_file + 'accuracy/' + dataname+'(k=' + str(k) + ')_accuracy_ZC_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataMV1 = pd.DataFrame({'budget': budget, 'accuracy': acc_MV})
    dataMV1.to_csv(output_file + 'accuracy/' +dataname+'(k=' + str(k) + ')_accuracy_MV_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataGLAD1 = pd.DataFrame({'budget': budget, 'accuracy': acc_GLAD})
    dataGLAD1.to_csv(output_file + 'accuracy/' + dataname+'(k=' + str(k) + ')_accuracy_GLAD_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataPM1 = pd.DataFrame({'budget': budget, 'accuracy': acc_PM})
    dataPM1.to_csv(output_file + 'accuracy/' + dataname+'(k=' + str(k) + ')_accuracy_PM_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataBCC1 = pd.DataFrame({'budget': budget, 'accuracy': acc_BCC})
    dataBCC1.to_csv(output_file + 'accuracy/' + dataname+'(k=' + str(k) + ')_accuracy_BCC_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataBWA1 = pd.DataFrame({'budget': budget, 'accuracy': acc_BWA})
    dataBWA1.to_csv(output_file + 'accuracy/' +dataname+'(k=' + str(k) + ')_accuracy_BWA_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    dataEBCC1 = pd.DataFrame({'budget': budget, 'accuracy': acc_EBCC})
    dataEBCC1.to_csv(output_file + 'accuracy/' +dataname+'(k=' + str(k) + ')_accuracy_EBCC_EBCC_' + str(deta) +'.txt', index=False, sep=',')

    # dataHC1 = pd.DataFrame({'budget': budget, 'accuracy_score': acc_HC})
    # dataHC1.to_csv(output_file + 'accuracy/' +dataname+'(k=' + str(k) + ')_accuracy_HC_EBCC_' + str(deta) +'.txt', index=False, sep=',')
