
#Importing/installing modules
import sys

# Core Libraries for Data Science
import pandas as pd
import numpy as np

# pre-processing
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, PredefinedSplit

# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.decomposition import PCA

# Evaluation & metrics
# from sklearn import metrics

# operations
from collections import Counter
import time
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pathlib import Path
import shutil

# ORIGINAL DATA LOADING:
def doggo_data_loading(seq_length=1, apply_seq_length=True):
    """
    seq_length - the min # of sequences a dog can have to be included in temporal data
    apply_seq_length - whether or not the seq_length is applied
    """
    
    # Data prep
    # pull up shortened data
    # pull up data for EDA
    heal_df = pd.read_csv("CSV_DATA/DAP_2020_HLES_health_condition_v1.1.csv")
    ownr_df = pd.read_csv("CSV_DATA/DAP_2020_HLES_dog_owner_v1.1.csv")
    cncr_df = pd.read_csv("CSV_DATA/DAP_2020_HLES_cancer_condition_v1.1.csv")

    # if in Google Colab:
    # heal_df = pd.read_csv("/content/drive/Shared drives/CAPSTONE/CSV_DATA/DAP_2020_HLES_health_condition_v1.1.csv")
    # ownr_df = pd.read_csv("/content/drive/Shared drives/CAPSTONE/CSV_DATA/DAP_2020_HLES_dog_owner_v1.1.csv")
    # cncr_df = pd.read_csv("/content/drive/Shared drives/CAPSTONE/CSV_DATA/DAP_2020_HLES_cancer_condition_v1.1.csv")

    ow_cols = ['dog_id','hs_health_conditions_cancer', # non-features
                  'hs_general_health','hs_recent_hospitalization', # cat 1
                  'db_excitement_level_before_walk','db_excitement_level_before_car_ride', # cat 2
                  'db_aggression_level_approached_while_eating','db_aggression_level_on_leash_unknown_dog', # cat 3
                  'db_fear_level_loud_noises','db_fear_level_unknown_human_touch', # cat 4
                  'db_left_alone_barking_frequency','db_left_alone_restlessness_frequency', # cat 5
                  'db_attention_seeking_follows_humans_frequency','db_attention_seeking_sits_close_to_humans_frequency', # cat 6
                  'db_training_distraction_frequency','db_training_obeys_stay_command_frequency', # cat 7
                'db_barks_frequency','db_escapes_home_or_property_frequency','db_playful_frequency',
               'db_urinates_in_home_frequency', # cat 8
                'dd_breed_pure_or_mixed','dd_breed_pure','dd_breed_pure_non_akc','dd_breed_mixed_primary', # cat 9
                  'dd_age_years','dd_spayed_or_neutered','dd_insurance','dd_sex','dd_weight_range', # cat 10
                  'de_daytime_sleep_avg_hours','de_nighttime_sleep_avg_hours','de_eats_feces',
               'de_eats_grass_frequency', # cat 11
               'df_appetite','df_appetite_change_last_year','df_ever_malnourished','df_ever_overweight','df_ever_underweight','df_feedings_per_day', # cat 12
                  'mp_dental_brushing_frequency','mp_flea_and_tick_treatment','mp_heartworm_preventative', # cat 13
                ]
    te_cols = ['dog_id','hs_condition_type','hs_diagnosis_month',
                     'hs_diagnosis_year','hs_follow_up_ongoing',
                     'hs_required_surgery_or_hospitalization']

    ca_cols = ['dog_id_cncr','hs_follow_up_ongoing_cncr', 'hs_initial_diagnosis_month_cncr', 
                   'hs_initial_diagnosis_year_cncr','hs_required_surgery_or_hospitalization_cncr']

    # need to rename cols with same names in DFs that will be joined
    cncr_df.rename(columns = {
        'dog_id':'dog_id_cncr', 'hs_follow_up_ongoing':'hs_follow_up_ongoing_cncr', 
        'hs_required_surgery_or_hospitalization':'hs_required_surgery_or_hospitalization_cncr', 
        'hs_initial_diagnosis_month':'hs_initial_diagnosis_month_cncr', 'hs_initial_diagnosis_year':'hs_initial_diagnosis_year_cncr'
        }, inplace = True)

    # for passing to data processor later
    ownr_df_sm = ownr_df[ow_cols]
    heal_df_sm = heal_df[te_cols]
    cncr_df_sm = cncr_df[ca_cols]
    
    ownr_cols = list(ownr_df_sm.columns)
    feat_list = ownr_cols
    
    temp_cols = list(heal_df_sm.columns)
    feat_list.extend(temp_cols)
    
    cncr_cols = list(cncr_df_sm.columns)
    feat_list.extend(cncr_cols)
    
    # join dataframes using doggo id
    joined_df = ownr_df_sm.set_index('dog_id').join(heal_df_sm.set_index('dog_id'))
    joined_df = joined_df.reset_index(level=0)
    joined_df = joined_df.set_index('dog_id').join(cncr_df_sm.set_index('dog_id_cncr'))
    joined_df = joined_df.reset_index(level=0)
    joined_df.rename(columns = {'index':'dog_id'}, inplace = True)
    
    # convert prediction columns to traditional bool:
    joined_df['hs_health_conditions_cancer'] = np.where(joined_df['hs_health_conditions_cancer'] == 2, 1, 0) 
    
    # Get rid of dogs that have no medical info (cancer or health entries)
    nan_cols = ['hs_condition_type','hs_diagnosis_month','hs_initial_diagnosis_month_cncr']
    joined_df = joined_df.dropna(subset=nan_cols, how='all')

    # set up state
    # state is 0 for all dogs with cancer=0
    # state is 0 for all dogs prior to the month/yr of cancer
    # stae is 1 for cancer entry and all preceding entries
    joined_df['state'] = 0
    # add the 1 state AFTER the iterrows below!

    joined_df['hs_condition_type'] = joined_df['hs_condition_type'].fillna(20) 

    # if there is not a separate cancer row, need to create (ONE PER ID)
    cur_dog_id = 0
    newrow_list = []
    for index, row in joined_df.iterrows():
      # check if row ID has changed AND if it is a dog with cancer indicated
      if cur_dog_id != row['dog_id']:
        if row['hs_health_conditions_cancer']==1:
            if row['hs_condition_type'] != 20:
                new_row = row
                # update rows in new row with cancer info
                new_row['hs_condition_type'] = 20
                new_row['hs_diagnosis_month'] = row['hs_initial_diagnosis_month_cncr']
                new_row['hs_diagnosis_year'] = row['hs_initial_diagnosis_year_cncr']
                # add the 1 state - during and after cancer entry, but not before!
                new_row['state'] = 1
                cncr_mo = new_row['hs_diagnosis_month']
                cncr_yr = new_row['hs_diagnosis_year']
                # add new row to list to be added to df
                newrow_list.append(new_row)

        cur_dog_id = row['dog_id']

    # add the 1 state - during and after cancer entry, but not before!
        # only for dogs with cancer
        # is the year = or > than cancr year? then check month
        # is the month = or > than cancer month? then state = 1
    # problem with separate yr and mo columns, so adding those together:
    joined_df['yrmo_combo'] = (joined_df['hs_diagnosis_year'] * 12) + joined_df['hs_diagnosis_month']
    joined_df['yrmo_combo_cncr'] = (joined_df['hs_initial_diagnosis_year_cncr'] * 12) + joined_df['hs_initial_diagnosis_month_cncr']
    # then do the conditions:
    conds = joined_df['hs_health_conditions_cancer'].eq(1) & joined_df['yrmo_combo'].ge(joined_df['yrmo_combo_cncr'])
    joined_df.loc[conds,'state']=1

    nr_df = pd.DataFrame(newrow_list)
    joined_df = pd.concat([joined_df, nr_df], ignore_index=True)
    joined_df = joined_df.reset_index(level=0)
      
    # turn cancer records into "medical" records (i.e. healthcare visit entry)
    # make cancer "condition_type"=20 (so new condition type)
    # based on these fields: 'hs_initial_diagnosis_month_cncr','hs_initial_diagnosis_year_cncr',
    # convert into the healthcare fields: 'hs_required_surgery_or_hospitalization_cncr'
    joined_df['hs_condition_type'] = joined_df['hs_condition_type'].fillna(20)  
    joined_df['hs_diagnosis_month'] = np.where(joined_df['hs_condition_type'] == 20, joined_df['hs_initial_diagnosis_month_cncr'], joined_df['hs_diagnosis_month'])
    joined_df['hs_diagnosis_year'] = np.where(joined_df['hs_condition_type'] == 20, joined_df['hs_initial_diagnosis_year_cncr'], joined_df['hs_diagnosis_year'])
    joined_df['hs_follow_up_ongoing'] = np.where(joined_df['hs_condition_type'] == 20, joined_df['hs_follow_up_ongoing_cncr'], joined_df['hs_follow_up_ongoing'])
    joined_df['hs_required_surgery_or_hospitalization'] = np.where(joined_df['hs_condition_type'] == 20, joined_df['hs_required_surgery_or_hospitalization_cncr'], joined_df['hs_required_surgery_or_hospitalization'])
    joined_df['state'] = np.where(joined_df['hs_condition_type'] == 20, 1, joined_df['state'])

    # Update dog breeds if that category selected (have to before dropping NaN)
    joined_df['dd_breed_pure'] = joined_df['dd_breed_pure'].fillna(0)
    joined_df['dd_breed_pure_non_akc'] = joined_df['dd_breed_pure_non_akc'].fillna(0)
    joined_df['dd_breed_mixed_primary'] = joined_df['dd_breed_mixed_primary'].fillna(0) 

    # add together for new
    temp_df = joined_df[['dd_breed_pure','dd_breed_pure_non_akc','dd_breed_mixed_primary']]
    sum_df = temp_df.sum(axis=1)
    joined_df['dd_combined_main_breed'] = sum_df
    feat_list.append('dd_combined_main_breed')

#     # remove old columns - can get rid of cancer fields at this point: 
    drop_cols = ['dd_breed_pure','dd_breed_pure_non_akc','dd_breed_mixed_primary', 'hs_initial_diagnosis_month_cncr',
                 'hs_initial_diagnosis_year_cncr', 'hs_follow_up_ongoing_cncr', 'hs_required_surgery_or_hospitalization_cncr',
                 'yrmo_combo_cncr','index', 'yrmo_combo']
    joined_df = joined_df.drop(columns=drop_cols) # , axis=1

    # remove rows that have NaN
    joined_df = joined_df.dropna()

    # remove these, update feat_num (only by 2 though, since adding a new one)
    feat_list.remove('dd_breed_pure')
    feat_list.remove('dd_breed_pure_non_akc')
    feat_list.remove('dd_breed_mixed_primary')
    feat_list.remove('hs_initial_diagnosis_month_cncr')
    feat_list.remove('hs_initial_diagnosis_year_cncr')
    feat_list.remove('hs_follow_up_ongoing_cncr')
    feat_list.remove('hs_required_surgery_or_hospitalization_cncr')
    feat_list.remove('dog_id_cncr')

    # re-sort - DOG IDs (either way, but grouped), then diag year (old to new - asscending), then diag month (old to new - asscending)
    joined_df = joined_df.sort_values(by=['dog_id', 'hs_diagnosis_year', 'hs_diagnosis_month'], ascending=[True,True,True], ignore_index=True)
    
    # reduce to only dogs with seq_lenth+ of records # turning off for now
    if apply_seq_length==True:
        joined_df = joined_df[joined_df.groupby('dog_id')['dog_id'].transform('size') >= seq_length]
    else: 
        joined_df = joined_df

    # GENERAL HEALTH'
    joined_df['hs_recent_hospitalization'] = joined_df['hs_recent_hospitalization'].astype(int)

    # OTHER DEMOGRAPHICS
    joined_df['dd_insurance'] = joined_df['dd_insurance'].astype(int)
    joined_df['dd_spayed_or_neutered'] = joined_df['dd_spayed_or_neutered'].astype(int)

    # ENVIRONMENTAL 
    joined_df['de_daytime_sleep_avg_hours'] = np.where(joined_df['de_daytime_sleep_avg_hours'] == 99, 0, 1)  
    joined_df['de_nighttime_sleep_avg_hours'] = np.where(joined_df['de_nighttime_sleep_avg_hours'] == 99, 0, 1)  
    joined_df['de_eats_feces'] = np.where(joined_df['de_eats_feces'] == 99, 0, 1)  

    # LIFESTYLE
    # if don't know one of these, going to assume a no
    joined_df['df_appetite_change_last_year'] = np.where(joined_df['df_appetite_change_last_year'] == 99, 0, joined_df['df_appetite_change_last_year'])
    joined_df['df_ever_malnourished'] = np.where(joined_df['df_ever_malnourished'] == 99, 0, joined_df['df_ever_malnourished'])
    joined_df['df_ever_overweight'] = np.where(joined_df['df_ever_overweight'] == 99, 0, joined_df['df_ever_overweight'])
    joined_df['df_ever_underweight'] = np.where(joined_df['df_ever_underweight'] == 99, 0, joined_df['df_ever_underweight'])

    # 'df_feedings_per_day', 1 (ONCE) 2 (TWICE) 3 (3+) 4 (Free fed) -> breakout free fed as separate data point 'free_fed'
    # break out the free fed:
    # set up bool field for t/f 1/0 'free_fed'
    joined_df['free_fed'] = (joined_df['df_feedings_per_day']==4).astype(int)
    joined_df['df_feedings_per_day'] = np.where(joined_df['df_feedings_per_day'] == 4, 1, joined_df['df_feedings_per_day'])
    # added a feature
    feat_list.append('free_fed')

    # MEDICATIONS / PREVENTIVES
    joined_df['mp_flea_and_tick_treatment'] = joined_df['mp_flea_and_tick_treatment'].astype(int) # 'mp_flea_and_tick_treatment', t/f
    joined_df['mp_heartworm_preventative'] = joined_df['mp_heartworm_preventative'].astype(int) # 'mp_heartworm_preventative', t/f

    # TEMPORAL
    # set follow up to be numeric, not t/f
    joined_df['hs_follow_up_ongoing'] = joined_df['hs_follow_up_ongoing'].astype(int)
    conds = joined_df['hs_required_surgery_or_hospitalization'].eq(1.0) # & joined_df['yrmo_combo'].ge(joined_df['yrmo_combo_cncr'])
    # update surgery status to be progressive 
    # want 'hs_required_surgery_or_hospitalization' to be more linear in nature (ie. 1 is nothing, 4 is worse (surgery AND hospitalization))
    joined_df['surg_hospital'] = 2
    joined_df['surg_hospital'] = np.where(joined_df['hs_required_surgery_or_hospitalization'] == 1, 3, joined_df['hs_required_surgery_or_hospitalization'])
    joined_df['surg_hospital'] = np.where(joined_df['hs_required_surgery_or_hospitalization'] == 3, 4, joined_df['hs_required_surgery_or_hospitalization'])  
    joined_df['surg_hospital'] = np.where(joined_df['hs_required_surgery_or_hospitalization'] == 4, 1, joined_df['hs_required_surgery_or_hospitalization'])  
    drop_cols = ['hs_required_surgery_or_hospitalization']
    joined_df = joined_df.drop(columns=drop_cols) # , axis=1
    feat_list.remove('hs_required_surgery_or_hospitalization')
    feat_list.append('surg_hospital')

    # final clean up:
    # take out the cancer field (data leakage and isn't static)
    feat_list.append('state')
    feat_list.remove('hs_health_conditions_cancer')
    feat_list.remove('dog_id')
    feat_list.remove('dog_id')
    feat_list.remove('hs_condition_type')
    feat_list.remove('hs_diagnosis_month')
    feat_list.remove('hs_diagnosis_year')
  
    # put state at end:
    cols = list(joined_df.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('state')) #Remove state from list
    joined_df = joined_df[cols+['state']] #Create new dataframe with columns in the order you want
    
    print('feature size: ', len(feat_list))
    print('Joined SIZE: ', joined_df.shape)
    
    return joined_df, feat_list  

# transform data to use in models with X and y
def getXy(df):
    # set up feature columns
    df_feats = df.columns.values.tolist()
    df_feats.remove('dog_id')
    df_feats.remove('hs_health_conditions_cancer')
    df_feats.remove('hs_condition_type')
    df_feats.remove('hs_diagnosis_month')
    df_feats.remove('hs_diagnosis_year')
    df_feats.remove('state')
    X = df[df_feats]
    y = df['state']

    return X, y

def get_default_dataset(seq_length=1):
    # Sets up DEFAULT Data Pattern - just using each row as an data point
    df, feat_list = doggo_data_loading(seq_length=seq_length)
    X, y = getXy(df)
    return X, y

def get_future_step_dataset(cat_feats = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], seq_length = 1, rand_state=0):
    # FUTURE STAGE Data Pattern - Using the step JUST BEFORE cancer develops as the "state=1",
    #  and then other rows are just dogs without cancer at all

    # find dog_ids that have hs_health_conditions_cancer==1, and get the last one that has value of 0 in the 'state' -> label this with pre_can=1
    #  -> can do this by getting and cancer=1 and state=0, and then just taking the last one for each ID!
    # get dog_ids that have hs_health_conditions_cancer==0, and laebl with pre_can=0
    # throw out rest of rows!
    df, feat_list = doggo_data_loading(seq_length=seq_length)

    precan_df = df[(df['hs_health_conditions_cancer']==1) & (df['state']==0)] # 4407 rows, 47 cols
    precan_df['state'] = 1
    nocan_df = df[(df['hs_health_conditions_cancer']==0)] # 64207 X 47
    nocan_df['state'] = 0

    # add a combined month/year value
    precan_df['yrmo_combo'] = (precan_df['hs_diagnosis_year'] * 12) + precan_df['hs_diagnosis_month']

    # get the highest yrmo_combo for each dog_id
    latest_entry = precan_df.groupby('dog_id').max('yrmo_combo')
    latest = precan_df.query('hs_health_conditions_cancer==1').groupby('dog_id').max('yrmo_combo').assign(Latest = "Latest")
    precan_df = pd.merge(precan_df,latest,how="outer")

    # NOTE: Keeps records that have same month/year, because both could indicate cancer
    precan_df = precan_df.dropna()

    drop_cols = ['yrmo_combo', 'Latest']
    precan_df = precan_df.drop(drop_cols, axis=1)

    frames = [precan_df, nocan_df]
    joined_df = pd.concat(frames)

    # get X and y based on new df
    X_fs, y_fs = getXy(joined_df)

    return X_fs, y_fs

# showing imbalanced data - set up SMOTENC to balance with synthesized oversampling
# summarize the original class distribution
def oversamp(X_values, y_values, sampling_strategy=0.5, cat_feats = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], rand_state=0):
    oversampler = SMOTENC(categorical_features=cat_feats, sampling_strategy=sampling_strategy, random_state=rand_state, n_jobs=-1)
    X_os, y_os = oversampler.fit_resample(X_values, y_values)
    print(f"Oversampling training set from {dict(Counter(y_values))} to {dict(Counter(y_os))}")
    return X_os, y_os

def undersamp(X_values, y_values, sampling_strategy=0.5, rand_state=0):
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=rand_state, replacement=False)
    X_us, y_us = undersampler.fit_resample(X_values, y_values)
    print(f"Undersampling training set from {dict(Counter(y_values))} to {dict(Counter(y_us))}")
    return X_us, y_us

def get_split_data(trainsplit=80, seqlen=1, oversample=False, oversample_strat=1.0, undersample=False, undersample_strat=0.5, future=False, rand_state=0):
    
    # trainsplit can be 50, 60, 70, or 80
    
    cat_feats = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

    X, y = None, None
    
    if future==True:
        X, y = get_future_step_dataset(cat_feats = cat_feats, seq_length=seqlen, rand_state=rand_state)
    else:
        X, y = get_default_dataset(seq_length=seqlen)

    # Default values for 60/20/20
    testp = 0.2
    valp = 0.25
    if trainsplit==80: # 80/10/10
        testp = 0.1
        valp = 0.115
    elif trainsplit==50: # 50/25/25
        testp = 0.25
        valp = 0.5
    elif trainsplit==70: # 70/20/10
        testp = 0.20
        valp = 0.15
    
    # For 50/25/25 split: S1 0.25, S2: 0.50
    # For 60/20/20 split: 0.2, 0.25
    # For 80/10/10 split: 0.1, 0.115
    
    # Split 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testp, stratify = y, random_state=rand_state)
    
    # Split 2
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valp, stratify = y_train, random_state=rand_state)
    
    # X_gs, y_gs = X, y
    
    if oversample==True:
        # now oversample / undersample it
        # 1.0 means it matches majority class
        X_oversampled, y_oversampled = oversamp(X_train, y_train, sampling_strategy=oversample_strat, cat_feats = cat_feats, rand_state=rand_state) 
        
        # Grid Search Data:
        # X_gs, y_gs = oversamp(X_gs, y_gs, sampling_strategy=oversample_strat, cat_feats = cat_feats, rand_state=rand_state) 
        
    if undersample==True:
        X_train, y_train = undersamp(X_oversampled, y_oversampled, sampling_strategy=undersample_strat, rand_state=rand_state)
        X_gs, y_gs = undersamp(X_gs, y_gs, sampling_strategy=undersample_strat, rand_state=rand_state)
    elif oversample==True:
        X_train, y_train = X_oversampled, y_oversampled
    
    # gs sets are train + test (validation still held out)
    
    X_gs, y_gs = (pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))  # X, y
    
    # split index is based on the gs set, and is the index of test=0, rest of index is -1
    test_fold = [0 if x in X_test.index else -1 for x in X_gs.index]
    
    # X_gs, y_gs, test_fold -> for grid search, test_fold goes into PredefinedSplit
    
    return X_gs, X_train, X_test, X_val, y_gs, y_train, y_test, y_val, test_fold