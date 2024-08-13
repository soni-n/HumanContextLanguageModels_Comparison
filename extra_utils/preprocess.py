import numpy as np
import pandas as pd

l0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TD0/labels.txt')
p0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TD0/preds.txt')
df0 = pd.DataFrame({'preds': p0, 'labels':l0})
df0['id'] = df0.index+1
df0
df0.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD0.csv', index=False)

hl0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TD0/labels.txt')
hp0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TD0/preds.txt')
hf0 = pd.DataFrame({'preds': hp0, 'labels':hl0})
hf0['id'] = hf0.index+1
hf0
hf0.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD0.csv', index=False)


python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD0.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD0.csv --is_csv True --sort_by id --pred_field preds --label_field labels



python /home/nisoni/eihart/EIHaRT/extra_utils/permutation_csv.py --pred_dir /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csv3/ --alt_dir /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csv3/


python /home/nisoni/eihart/EIHaRT/extra_utils/permutation_csv.py --pred_dir /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csv2/ --alt_dir /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csv2/


 python /home/nisoni/eihart/EIHaRT/extra_utils/permutation_csv.py --pred_dir /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/ --alt_dir /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/


l2 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TD2/labels.txt')
p2 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TD2/preds.txt')
df2 = pd.DataFrame({'preds': p2, 'labels':l2})
df2['id'] = df2.index+1
df2
df2.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD2.csv', index=False)

hl2 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TD2/labels.txt')
hp2 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TD2/preds.txt')
hf2 = pd.DataFrame({'preds': hp2, 'labels':hl2})
hf2['id'] = hf2.index+1
hf2
hf2.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD2.csv', index=False)

python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD2.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD2.csv --is_csv True --sort_by id --pred_field preds --label_field labels


l3 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TP_TD_Age/labels.txt')
p3 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_tpft_after_trials_final/TP_eval/TP_TD_Age/preds.txt')
df3 = pd.DataFrame({'preds': p3, 'labels':l3})
df3['id'] = df3.index+1
df3
df3.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD3.csv', index=False)

hl3 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TP_TD_Age/labels.txt')
hp3 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_final/HaRT_downstream_final/TP_eval/TP_TD_Age/preds.txt')
hf3 = pd.DataFrame({'preds': hp3, 'labels':hl3})
hf3['id'] = hf3.index+1
hf3
hf3.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD3.csv', index=False)

python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_final/hart_preds/csvs/TD3.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_final/gpt2hlc_preds/csvs/TD3.csv --is_csv True --sort_by id --pred_field preds --label_field labels



############# FB age #########################


### eihart_age
l0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_age4bl_lastepoch_lbpeval/labels.txt')
p0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_age4bl_lastepoch_lbpeval/preds.txt')
df0 = pd.DataFrame({'preds': p0, 'labels':l0})
df0['id'] = df0.index+1
df0
df0.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/eihart_age.csv', index=False)

### eihart_ope
l00 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_OPE4blocks_to_Age4bl/continue_PT_lbp_eval/labels.txt')
p00 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_OPE4blocks_to_Age4bl/continue_PT_lbp_eval/preds.txt')
df00 = pd.DataFrame({'preds': p00, 'labels':l00})
df00['id'] = df00.index+1
df00
df00.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/eihart_ope.csv', index=False)


#### hart age
hl0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/HaRT_preds/age/labels.txt')
hp0 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/HaRT_preds/age/preds.txt')
hf0 = pd.DataFrame({'preds': hp0, 'labels':hl0})
hf0['id'] = hf0.index+1
hf0
hf0.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/hart.csv', index=False)


python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/eihart_age.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/hart.csv --is_csv True --sort_by id --pred_field preds --label_field labels

python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/eihart_ope.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_age/hart.csv --is_csv True --sort_by id --pred_field preds --label_field labels




############# FB ope #########################


### eihart_age
l1 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_age4blocks_to_ope4bl/continue_PT_lbp_eval/labels.txt')
p1 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_age4blocks_to_ope4bl/continue_PT_lbp_eval/preds.txt')
df1 = pd.DataFrame({'preds': p1, 'labels':l1})
df1['id'] = df1.index+1
df1
df1.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/eihart_age.csv', index=False)

### eihart_ope
l10 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_ope4bl_lbpeval/labels.txt')
p10 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/eihart_ope4bl_lbpeval/preds.txt')
df10 = pd.DataFrame({'preds': p10, 'labels':l10})
df10['id'] = df10.index+1
df10
df10.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/eihart_ope.csv', index=False)

#### hart age
hl1 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/HaRT_preds/ope/full/labels.txt')
hp1 = np.loadtxt('/chronos_data/nisoni/EIHaRT_Downstream_user_final/HaRT_preds/ope/full/preds.txt')
hf1 = pd.DataFrame({'preds': hp1, 'labels':hl1})
hf1['id'] = hf1.index+1
hf1
hf1.to_csv('/chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/hart.csv', index=False)


python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/eihart_age.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/hart.csv --is_csv True --sort_by id --pred_field preds --label_field labels

python /home/nisoni/eihart/EIHaRT/extra_utils/paired_ttest.py --t1 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/eihart_ope.csv --t2 /chronos_data/nisoni/EIHaRT_Downstream_user_final/significance_test/FB_ope/hart.csv --is_csv True --sort_by id --pred_field preds --label_field labels