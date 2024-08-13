import pandas as pd
import numpy as np

age = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/age.pkl')
age_df = pd.DataFrame.from_dict(age, orient='index')
age_df["user_id"] = age_df.index
age_df['user_id'] = age_df['user_id'].astype(int)
age_df.rename(columns = {0: 'inferred_age'}, inplace=True)
age_df.reset_index(drop=True, inplace=True)


####### abo test #####
abo_test = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance/abo_test_all.pkl')
abo_test_wage = abo_test.merge(age_df, on='user_id')
assert  len(np.where(abo_test_wage['inferred_age'] > 0 )[0]) == abo_test.shape[0]

abo_test_less25 = abo_test_wage[abo_test_wage['inferred_age'] < 25 ] #(89,8)
abo_test_25toless35 = abo_test_wage[(abo_test_wage['inferred_age'] >= 25) & (abo_test_wage['inferred_age'] < 35)]  #(429,8)
abo_test_35to44 = abo_test_wage[(abo_test_wage['inferred_age'] >= 35) & (abo_test_wage['inferred_age'] < 45)]  #(108,8)
abo_test_45andmore = abo_test_wage[abo_test_wage['inferred_age'] >= 45 ] #(5,8)

abo_test_less25.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/abo/abo_test_less25.pkl')
abo_test_25toless35.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/abo/abo_test_25toless35.pkl')
abo_test_35to44.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/abo/abo_test_35to44.pkl')
abo_test_45andmore.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/abo/abo_test_45andmore.pkl')

####### ath test #####
ath_test = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance/ath_test_all.pkl')
ath_test_wage = ath_test.merge(age_df, on='user_id')
assert  len(np.where(ath_test_wage['inferred_age'] > 0 )[0]) == ath_test.shape[0]

ath_test_less25 = ath_test_wage[ath_test_wage['inferred_age'] < 25 ] #(1278,8)
ath_test_25toless35 = ath_test_wage[(ath_test_wage['inferred_age'] >= 25) & (ath_test_wage['inferred_age'] < 35)]  #(384,8)
ath_test_35to44 = ath_test_wage[(ath_test_wage['inferred_age'] >= 35) & (ath_test_wage['inferred_age'] < 45)]  #(25,8)
ath_test_45andmore = ath_test_wage[ath_test_wage['inferred_age'] >= 45 ] #(1,8)

ath_test_less25.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/ath/ath_test_less25.pkl')
ath_test_25toless35.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/ath/ath_test_25toless35.pkl')
ath_test_35to44.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/ath/ath_test_35to44.pkl')
ath_test_45andmore.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/ath/ath_test_45andmore.pkl')

####### clim test #####
clim_test = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance/clim_test_all.pkl')
clim_test_wage = clim_test.merge(age_df, on='user_id')
assert len(np.where(clim_test_wage['inferred_age'] > 0 )[0]) == clim_test.shape[0]

clim_test_less25 = clim_test_wage[clim_test_wage['inferred_age'] < 25 ] #(22,8)
clim_test_25toless35 = clim_test_wage[(clim_test_wage['inferred_age'] >= 25) & (clim_test_wage['inferred_age'] < 35)]  #(581,8)
clim_test_35to44 = clim_test_wage[(clim_test_wage['inferred_age'] >= 35) & (clim_test_wage['inferred_age'] < 45)]  #(132,8)
clim_test_45andmore = clim_test_wage[clim_test_wage['inferred_age'] >= 45 ] #(5,8)

clim_test_less25.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clim/clim_test_less25.pkl')
clim_test_25toless35.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clim/clim_test_25toless35.pkl')
clim_test_35to44.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clim/clim_test_35to44.pkl')
clim_test_45andmore.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clim/clim_test_45andmore.pkl')

####### clin test #####
clin_test = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance/clin_test_all.pkl')
clin_test_wage = clin_test.merge(age_df, on='user_id')
assert len(np.where(clin_test_wage['inferred_age'] > 0 )[0]) == clin_test.shape[0]

clin_test_less25 = clin_test_wage[clin_test_wage['inferred_age'] < 25 ] #(29,8)
clin_test_25toless35 = clin_test_wage[(clin_test_wage['inferred_age'] >= 25) & (clin_test_wage['inferred_age'] < 35)]  #(496,8)
clin_test_35to44 = clin_test_wage[(clin_test_wage['inferred_age'] >= 35) & (clin_test_wage['inferred_age'] < 45)]  #(24,8)
clin_test_45andmore = clin_test_wage[clin_test_wage['inferred_age'] >= 45 ] #(8,8)

clin_test_less25.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clin/clin_test_less25.pkl')
clin_test_25toless35.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clin/clin_test_25toless35.pkl')
clin_test_35to44.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clin/clin_test_35to44.pkl')
clin_test_45andmore.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/clin/clin_test_45andmore.pkl')

####### fem test #####
fem_test = pd.read_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance/fem_test_all.pkl')
fem_test_wage = fem_test.merge(age_df, on='user_id')
assert len(np.where(fem_test_wage['inferred_age'] > 0 )[0]) == fem_test.shape[0]

fem_test_less25 = fem_test_wage[fem_test_wage['inferred_age'] < 25 ] #(297,8)
fem_test_25toless35 = fem_test_wage[(fem_test_wage['inferred_age'] >= 25) & (fem_test_wage['inferred_age'] < 35)]  #(174,8)
fem_test_35to44 = fem_test_wage[(fem_test_wage['inferred_age'] >= 35) & (fem_test_wage['inferred_age'] < 45)]  #(10,8)

fem_test_less25.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/fem/fem_test_less25.pkl')
fem_test_25toless35.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/fem/fem_test_25toless35.pkl')
fem_test_35to44.to_pickle('/home/nisoni/eihart/EIHaRT/data/datasets/stance_test_agebuckets/fem/fem_test_35to44.pkl')