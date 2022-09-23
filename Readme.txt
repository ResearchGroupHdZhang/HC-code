./crowdsourcing_processor：processor类，任务处理
./dataloader：数据存储器，包括正常迭代NormalDataloader，以及熵增迭代IncreEntroDataloader
./fact：factset类
./output：Time/Sum_Entropy_结果(txt)
./output_post_p_txt：原始数据经过crowdsourcing后的post_p结果(txt)
./outputcsv：Time/Sum_Entropy_结果(csv)
./outputpdf：Time/Sum_Entropy_结果(pdf图)
./post_p_csv：原始数据经过crowdsourcing后的post_p结果(csv)
./query：query类，fact选择，包括Base(暴力法)、Greedy(近似算法)、Random(随机)
./resource：数据集(final_rte_dataset.xlsx, rte_final1.txt)
./worker：worker类
-----------------------------------------------------------------------------------------------------------
.onetask_21fact_时间对比.csv：暴力法与近似算法时间对比运行结果
.create_post_p_csv.py：From ./output_post_p_txt to ./outputcsv
.outputcsv_to_outputtxt.py：From ./outputcsv to ./output
.plot：
.xlsxtotxt：From final_rte_dataset.xlsx to rte_final1.txt
-----------------------------------------------------------------------------------------------------------
俩种结果的流程：
Time_enprory图：main.py生成./outputcsv --[outputcsv_to_outputtxt.py]--> ./output --[plot.py]--> 图 ./outputpdf
cr后的数据集：main.py生成./output_post_p_txt --[create_post_p_csv.py]--> ./post_p_csv