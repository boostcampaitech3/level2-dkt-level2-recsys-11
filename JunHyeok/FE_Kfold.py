# %%
import feature_engineering as Fe
import pandas as pd
import os
import random
import math
from tqdm import tqdm
import numpy as np
import time

# %%
dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

# 데이터 경로 맞춰주세요!
DATA_PATH = '/opt/ml/input/data/train_data.csv'
df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True) 

# %%
# dtype = {
#     'userID': 'int16',
#     'answerCode': 'int8',
#     'KnowledgeTag': 'int16'
# }   

# train = pd.read_csv("/opt/ml/input/data/cv_train_data.csv", dtype=dtype, parse_dates=['Timestamp'])
# valid = pd.read_csv("/opt/ml/input/data/cv_valid_data.csv", dtype=dtype, parse_dates=['Timestamp'])

# %%
dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

# 데이터 경로 맞춰주세요!
test_csv_file_path = '/opt/ml/input/data/test_data.csv'
test = pd.read_csv(test_csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
test = test.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

# %%
# 문제 번호
df['problem_number'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))
test['problem_number'] = test['assessmentItemID'].apply(lambda x : int(x[-3:]))

# %%
correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
correct_t.columns = ["test_mean", 'test_sum']

correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
correct_k.columns = ["tag_mean", 'tag_sum']

correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
correct_a.columns = ["ass_mean", 'ass_sum']

correct_p = df.groupby(['problem_number'])['answerCode'].agg(['mean', 'sum'])
correct_p.columns = ["prb_mean", 'prb_sum']

# %%
prac = df.copy()

# %% [markdown]
# # FE

# %%
prac = Fe.IK_question_acc(prac)
prac = Fe.IK_KnowledgeTag_acc(prac)
test = Fe.IK_question_acc(test)
test = Fe.IK_KnowledgeTag_acc(test)

# %%
# #time elapsed 
# diff = prac.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
# diff = diff.fillna(pd.Timedelta(seconds=0))
# diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
# prac['elapsed'] = diff
# prac['elapsed'] = prac['elapsed'].apply(lambda x : x if x <650 and x >=0 else 0)

# %%
#solved_question , 유저별 푼 문제 수
prac["solved_question"] = prac.groupby(["userID"]).cumcount()
test["solved_question"] = test.groupby(["userID"]).cumcount()

# %%
#time elapsed 
def change_elapsed(data):
    return data["elapsed"] if not data["is_elapsed_more_600"] else data["userID_testId_elapsed_mean"]
    
def change_log_elapsed(data):
    return data["log_elapsed"] if not data["is_elapsed_more_10"] else data["userID_testId_log_elapsed_mean"]

def get_elapsed(df):
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=0))
    elapsed = diff['Timestamp'].apply(lambda x: x.total_seconds())
    elapsed_log = diff['Timestamp'].apply(lambda x: np.log1p(x.total_seconds()))
    
    df['elapsed'] = elapsed
    df['log_elapsed'] = elapsed_log
    
    ### 특정 기준치를 벗어나면 해당 user의 해당 test지에 대한 평균 elapsed time으로 교체
    df["is_elapsed_more_600"] = df["elapsed"].apply(lambda x : True if x > 600 else False)
    df["is_elapsed_more_10"] = df["log_elapsed"].apply(lambda x : True if x > 10 else False)
    
    group_userID_testId = df[~df["is_elapsed_more_600"]].groupby(["userID", "testId"])["elapsed"].agg(["mean"])
    group_userID_testId = group_userID_testId.reset_index()
    group_userID_testId.columns = ["userID", "testId", "userID_testId_elapsed_mean"]

    group_userID_testId_log = df[~df["is_elapsed_more_10"]].groupby(["userID", "testId"])["log_elapsed"].agg(["mean"])
    group_userID_testId_log = group_userID_testId_log.reset_index()
    group_userID_testId_log.columns = ["userID", "testId", "userID_testId_log_elapsed_mean"]

    df = pd.merge(df, group_userID_testId, on=["userID", "testId"], how="left")
    df = pd.merge(df, group_userID_testId_log, on=["userID", "testId"], how="left")

    df["elapsed"] = df.apply(change_elapsed, axis=1)
    df["log_elapsed"] = df.apply(change_log_elapsed, axis=1)
    
    df = df.drop(['is_elapsed_more_600', 'is_elapsed_more_10', 'userID_testId_elapsed_mean', 'userID_testId_log_elapsed_mean'], axis=1)
    df = df.fillna(0.0)
    
    return df

# %%
prac = get_elapsed(prac)
test = get_elapsed(test)

# %%
def is_probably_easy(row):
    delta = row.delta
    delta_thres = 1 # hour
    
    is_prev_ord = row.is_previous_ordered
    is_prev_dec = row.is_previous_decreasing
    is_prev_ord_shift = row.is_prev_ord_shift
    is_prev_dec_shift = row.is_prev_dec_shift
    
    case = (is_prev_ord_shift, is_prev_dec_shift, is_prev_ord, is_prev_dec)
    
    probably_easy_l = [
        (np.nan, np.nan, -1, -1),
        (-1, -1, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 0),
    ]
    
    if pd.isnull(delta) or delta > pd.Timedelta(hours=1):
        return -1
    elif case in probably_easy_l:
        return 1
    else:
        return 0

def is_previous_decreasing(row):
    q_num = row.problem_number
    q_num_prev = row.q_num_prev
    delta = row.delta
    delta_thres = 1 # hour
    
    if pd.isnull(delta) or delta > pd.Timedelta(hours=1):
        return -1
    elif q_num < q_num_prev:
        return 1
    else:
        return 0

def add_last_problem(df):
    new = []
    pre = df['testId'][0]
    for idx in df['testId']:
        if pre != idx :
            new[-1]=-1
            pre = idx
        new.append(0)
    df['last_problem'] = new
    return df

def is_previous_ordered(row):
    q_num = row.problem_number
    q_num_prev = row.q_num_prev
    delta = row.delta
    delta_thres = 1 # hour
    
    if pd.isnull(delta) or delta > pd.Timedelta(hours=1):
        return -1
    elif q_num == q_num_prev + 1:
        return 1
    else:
        return 0

# %%
def feature_engineering(df):
    print('-'*20, 'Feature Engineering Start', '-'*20)
    start_time = time.time()
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    df = add_last_problem(df)
    # elo 추가
    # df = ELO_function(df)
    
    df['hour'] = df['Timestamp'].dt.hour
    df['dow'] = df['Timestamp'].dt.dayofweek
    
    # 푸는 시간
    # diff = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    # diff = diff.fillna(pd.Timedelta(seconds=0))
    # diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    # df['elapsed'] = diff
    # df['elapsed'] = df['elapsed'].apply(lambda x : x if x <650 and x >=0 else 0)
    
    df['grade']=df['testId'].apply(lambda x : int(x[1:4])//10)
    df['mid'] = df['testId'].apply(lambda x : int(x[-3:]))
    df['problem_number'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))
    
#     stu_test_groupby = df.groupby(['userID', 'testId'])
#     df.loc[:, "delta"] = stu_test_groupby['Timestamp'].diff()
#     df['q_num_prev'] = df.problem_number.shift()
#     df['is_previous_ordered'] =  df.apply(lambda row: is_previous_ordered(row), axis=1)
#     df['is_previous_decreasing'] = df.apply(lambda row: is_previous_decreasing(row), axis=1)
#     df['is_prev_ord_shift'] = df.is_previous_ordered.shift()
#     df['is_prev_dec_shift'] = df.is_previous_decreasing.shift()
#     df['is_probably_easy'] = df.apply(lambda row: is_probably_easy(row), axis=1)
#     df.drop(labels=['delta', 'q_num_prev', 'is_previous_ordered',
#                     'is_previous_decreasing', 'is_prev_ord_shift', 'is_prev_dec_shift'], axis=1, inplace=True)
    
    correct_h = df.groupby(['hour'])['answerCode'].agg(['mean', 'sum'])
    correct_h.columns = ["hour_mean", 'hour_sum']
    correct_d = df.groupby(['dow'])['answerCode'].agg(['mean', 'sum'])
    correct_d.columns = ["dow_mean", 'dow_sum'] 
    
    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
    df = pd.merge(df, correct_p, on=['problem_number'], how="left")
    df = pd.merge(df, correct_h, on=['hour'], how="left")
    df = pd.merge(df, correct_d, on=['dow'], how="left")

    o_df = df[df['answerCode']==1]
    x_df = df[df['answerCode']==0]
    
    elp_k = df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k.columns = ['KnowledgeTag',"tag_elp"]
    elp_k_o = o_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k_o.columns = ['KnowledgeTag', "tag_elp_o"]
    elp_k_x = x_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
    elp_k_x.columns = ['KnowledgeTag', "tag_elp_x"]
    
    df = pd.merge(df, elp_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_o, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_x, on=['KnowledgeTag'], how="left")

    ass_k = df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k.columns = ['assessmentItemID',"ass_elp"]
    ass_k_o = o_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k_o.columns = ['assessmentItemID',"ass_elp_o"]
    ass_k_x = x_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
    ass_k_x.columns = ['assessmentItemID',"ass_elp_x"]

    df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

    prb_k = df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k.columns = ['problem_number',"prb_elp"]
    prb_k_o = o_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k_o.columns = ['problem_number',"prb_elp_o"]
    prb_k_x = x_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
    prb_k_x.columns = ['problem_number',"prb_elp_x"]

    df = pd.merge(df, prb_k, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_o, on=['problem_number'], how="left")
    df = pd.merge(df, prb_k_x, on=['problem_number'], how="left")
    
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = (df['user_correct_answer']/df['user_total_answer']).fillna(0)
    df['Grade_o'] = df.groupby(['userID','grade'])['answerCode'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    df['GradeCount'] = df.groupby(['userID','grade']).cumcount()
    df['GradeAcc'] = (df['Grade_o']/df['GradeCount']).fillna(0)
    df['GradeElp'] = df.groupby(['userID','grade'])['elapsed'].transform(lambda x: x.cumsum()).fillna(0)
    df['GradeMElp'] = df['GradeElp']/[v if v != 0 else 1 for v in df['GradeCount'].values]
    
    f = lambda x : len(set(x))
    test = df.groupby(['testId']).agg({
        'problem_number':'max',
        'KnowledgeTag':f
    })
    test.reset_index(inplace=True)

    test.columns = ['testId','problem_count',"tag_count"]
    
    df = pd.merge(df,test,on='testId',how='left')
    
    gdf = df[['userID','testId','problem_number','grade','Timestamp']].sort_values(by=['userID','grade','Timestamp'])
    gdf['buserID'] = gdf['userID'] != gdf['userID'].shift(1)
    gdf['bgrade'] = gdf['grade'] != gdf['grade'].shift(1)
    gdf['first'] = gdf[['buserID','bgrade']].any(axis=1).apply(lambda x : 1- int(x))
    gdf['RepeatedTime'] = gdf['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)) 
    gdf['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x: x.total_seconds()) * gdf['first']
    df['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x : math.log(x+1))
    
    df['prior_KnowledgeTag_frequency'] = df.groupby(['userID','KnowledgeTag']).cumcount()
    
    df['problem_position'] = df['problem_number'] / df["problem_count"]
    df['solve_order'] = df.groupby(['userID','testId']).cumcount()
    df['solve_order'] = df['solve_order'] - df['problem_count']*(df['solve_order'] > df['problem_count']).apply(int) + 1
    df['retest'] = (df['solve_order'] > df['problem_count']).apply(int)
    T = df['solve_order'] != df['problem_number']
    TT = T.shift(1)
    TT[0] = False
    df['solved_disorder'] = (TT.apply(lambda x : not x) & T).apply(int)
    
    df['testId'] = df['testId'].apply(lambda x : int(x[1:4]+x[-3]))
    
    print('-'*20, 'Feature Engineering End', '-'*20)
    print(f"Feature Engineering에 걸린 시간 : {time.time() - start_time}s")
    return df

# %%
prac = feature_engineering(prac)
prac.head()

# %%
test = feature_engineering(test)
test.head()

# %%
# 얘도 이미 나눠져있어서 필요없음
# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
random.seed(42)
n_kfold = 5
def custom_kfold_split(df, kfold=n_kfold, split=True):
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    each_train_data_len = len(df) // 5
    sum_of_train_data = 0
    # user_ids =[]
    user_ids =[[] for _ in range(kfold)]

    k=0
    for user_id, count in users:
        sum_of_train_data += count
        if each_train_data_len * (k+1) < sum_of_train_data:
            k+=1
        if k==kfold:
            break
        user_ids[k].append(user_id)

    return user_ids

# %%
def mk_train_valid_dataset(user_id):
    train = prac[prac['userID'].isin(user_id)]
    valid = train[train['userID'] != train['userID'].shift(-1)]
    
    train = train.fillna(0)
    valid = valid.fillna(0)
    
    return train, valid

# %%
pieces = custom_kfold_split(prac, kfold=5) 

# %%
trains, valids = [], []

for i in pieces:
    _train, _valid = mk_train_valid_dataset(i)
    trains.append(_train)
    valids.append(_valid)


# %%
save_root = '/opt/ml/input/data/FE_dataset/kfold'

for i, (train, valid) in enumerate(zip(trains, valids)):
    train.to_csv(save_root + f'/train_after_{i}.csv', index = False)
    valid.to_csv(save_root + f'/valid_after_{i}.csv', index = False)
test.to_csv(save_root + '/test_after.csv', index = False)

# %% [markdown]
# # 카테고리 피처 라벨링

# %%
# 유저별 분리

# 사용할 Feature 설정
FEATS = ['testId', 
       'KnowledgeTag', 'problem_number', 'IK_question_acc',
       'IK_KnowledgeTag_acc', 'solved_question', 'elapsed', 'log_elapsed',
       'last_problem', 'hour', 'dow', 'grade', 'mid', 'test_mean', 'test_sum',
       'tag_mean', 'tag_sum', 'ass_mean', 'ass_sum', 'prb_mean', 'prb_sum',
       'hour_mean', 'hour_sum', 'dow_mean', 'dow_sum', 'tag_elp', 'tag_elp_o',
       'tag_elp_x', 'ass_elp', 'ass_elp_o', 'ass_elp_x', 'prb_elp',
       'prb_elp_o', 'prb_elp_x', 'user_correct_answer', 'user_total_answer',
       'user_acc', 'Grade_o', 'GradeCount', 'GradeAcc', 'GradeElp',
       'GradeMElp', 'problem_count', 'tag_count', 'RepeatedTime',
       'prior_KnowledgeTag_frequency', 'problem_position', 'solve_order',
       'retest', 'solved_disorder']



# %%
# train = pd.read_csv(save_root + "/train_after.csv")
# valid = pd.read_csv(save_root + "/valid_after.csv")

# # %%
# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ordinal_feats = ['grade']
# label_feats = ['problem_number','hour','dow','solved_disorder','KnowledgeTag','testId','retest']
        
# # 'problem_number','grade', 'testId','KnowledgeTag','problem_count','type_count','solved_disorder'

# for c in ordinal_feats :
#     X = train[c].values.reshape(-1,1)
#     enc = OrdinalEncoder()
#     enc.fit(X)
#     X = enc.transform(X)
#     train[c] = X

#     X = valid[c].values.reshape(-1,1)
#     X = enc.transform(X)
#     valid[c] = X
    
#     X = test[c].values.reshape(-1,1)
#     X = enc.transform(X)
#     test[c] = X
    
# for c in label_feats :
#     X = train[c].values.reshape(-1,1)
#     enc = LabelEncoder()
#     enc.fit(X)
#     X = enc.transform(X)
#     train[c] = X

#     X = valid[c].values.reshape(-1,1)
#     X = enc.transform(X)
#     valid[c] = X
    
#     X = test[c].values.reshape(-1,1)
#     X = enc.transform(X)
#     test[c] = X

# # %%
# # X, y 값 분리
# y_train = train['answerCode']
# train = train.drop(['answerCode'], axis=1)

# y_valid = valid['answerCode']
# valid = valid.drop(['answerCode'], axis=1)



# # %%
# import lightgbm as lgb

# lgb_train = lgb.Dataset(train[FEATS], y_train)
# lgb_valid = lgb.Dataset(valid[FEATS], y_valid)

# # %% [markdown]
# # # Training

# # %%
# import lightgbm as lgb
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score
# import numpy as np

# # %%
# '''
#     original
# '''
# model = lgb.train(
#                     {'objective': 'binary','metric':'auc', 'boosting' : 'dart', 'learning_rate':0.05},
# #                     {'objective': 'binary','metric':'auc','boosting':'dart',
# #                      'learning_rate':0.05,'max_depth':15,'feature_fraction':0.8},
#                     lgb_train,
#                     valid_sets=[lgb_train, lgb_valid],
#                     verbose_eval=100,
#                     num_boost_round=8000,
#                     early_stopping_rounds=200
#                 )

# # lgb_test = lgb.Dataset(test[FEATS], y_test)
# # model = lgb.train(
# #     {'objective': 'binary'}, 
# #     lgb_train,
# #     valid_sets=[lgb_train, lgb_test],
# #     verbose_eval=100,
# #     num_boost_round=500,
# #     early_stopping_rounds=100
# # )

# model.save_model('/opt/ml/output/LGBM_4.txt')

# preds = model.predict(valid[FEATS])
# acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
# auc = roc_auc_score(y_valid, preds)

# print(f'VALID AUC : {auc} ACC : {acc}\n')
# #VALID AUC : 0.8352326817779808 ACC : 0.757847533632287 - LGBM2 8000round 두번째 낸거
# #VALID AUC : 0.8290278343651559 ACC : 0.7548579970104634 - LGBM3 2000round 처음낸거

# %%
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

total_ind = list(range(n_kfold))

for i in range(n_kfold):
    
    train = [pd.read_csv(save_root + f"/train_after_{j}.csv") for j in total_ind if j!=i]
    train = pd.concat(train)
    valid = pd.read_csv(save_root + f"/valid_after_{i}.csv")


    ordinal_feats = ['grade']
    label_feats = ['problem_number','hour','dow','solved_disorder','KnowledgeTag','testId','retest']
            
    # 'problem_number','grade', 'testId','KnowledgeTag','problem_count','type_count','solved_disorder'

    for c in ordinal_feats :
        X = train[c].values.reshape(-1,1)
        enc = OrdinalEncoder()
        enc.fit(X)
        X = enc.transform(X)
        train[c] = X

        X = valid[c].values.reshape(-1,1)
        X = enc.transform(X)
        valid[c] = X
        
        # X = test[c].values.reshape(-1,1)
        # X = enc.transform(X)
        # test[c] = X
        
    for c in label_feats :
        X = train[c].values.reshape(-1,1)
        enc = LabelEncoder()
        enc.fit(X)
        X = enc.transform(X)
        train[c] = X

        X = valid[c].values.reshape(-1,1)
        X = enc.transform(X)
        valid[c] = X
        
        # X = test[c].values.reshape(-1,1)
        # X = enc.transform(X)
        # test[c] = X


    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_valid = valid['answerCode']
    valid = valid.drop(['answerCode'], axis=1)


    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_valid = lgb.Dataset(valid[FEATS], y_valid)


    '''
        original
    '''
    model = lgb.train(
                        {'objective': 'binary','metric':'auc', 'boosting' : 'dart', 'learning_rate':0.05},
    #                     {'objective': 'binary','metric':'auc','boosting':'dart',
    #                      'learning_rate':0.05,'max_depth':15,'feature_fraction':0.8},
                        lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        verbose_eval=100,
                        num_boost_round=8000,
                        early_stopping_rounds=200
                    )

    # lgb_test = lgb.Dataset(test[FEATS], y_test)
    # model = lgb.train(
    #     {'objective': 'binary'}, 
    #     lgb_train,
    #     valid_sets=[lgb_train, lgb_test],
    #     verbose_eval=100,
    #     num_boost_round=500,
    #     early_stopping_rounds=100
    # )

    model.save_model(f'/opt/ml/output/kfold/LGBM_{i}.txt')

    preds = model.predict(valid[FEATS])
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    #VALID AUC : 0.8352326817779808 ACC : 0.757847533632287 - LGBM2 8000round 두번째 낸거
    #VALID AUC : 0.8290278343651559 ACC : 0.7548579970104634 - LGBM3 2000round 처음낸거




# %% [markdown]
# # INFERENCE

# %%
import pandas as pd

test = pd.read_csv(save_root + '/test_after.csv')
test = test.fillna(0)
# LEAVE LAST INTERACTION ONLY
test = test[test['userID'] != test['userID'].shift(-1)]
# DROP ANSWERCODE
y_test = test['answerCode']
test = test.drop(['answerCode'], axis=1)

# %%
test_last = pd.read_csv(save_root + '/test_after.csv')
test_last = test_last.fillna(0)
test_last = test_last[~test_last.index.isin(test.index)]
# DROP ANSWERCODE
y_test_last = test_last['answerCode']
test_last = test_last.drop(['answerCode'], axis=1)

# %%
# rows 744인거 확인하기
test
# %%
# MAKE PREDICTION
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

output_root = '/opt/ml/output/kfold'
save_root = '/opt/ml/input/data/FE_dataset/kfold'

for i in range(n_kfold):
    model = lgb.Booster(model_file=f'{output_root}/LGBM_{i}.txt')
    total_preds = model.predict(test_last[FEATS])
    acc = accuracy_score(y_test_last, np.where(total_preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test_last, total_preds)
    print(f"acc_{i}\t{acc}")
    print(f"auc_{i}\t{auc}")

# %%
# 모델 선택
i = 1
model = lgb.Booster(model_file=f'{output_root}/LGBM_{i}.txt')
total_preds = model.predict(test[FEATS])

# %%
# SAVE OUTPUT
output_dir = '/opt/ml/output/'
write_path = os.path.join(output_dir, "LGBM4.csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)    
with open(write_path, 'w', encoding='utf8') as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(total_preds):
        w.write('{},{}\n'.format(id,p))

# %%
len(total_preds)

# %%



