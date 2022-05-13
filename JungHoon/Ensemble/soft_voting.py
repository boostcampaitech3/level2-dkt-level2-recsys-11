# %%
import numpy as np
import pandas as pd


# %%
suffix = ''
file_list = ["output_ATTNnew.csv", "output_bert.csv", "output_lgbm.csv", "output_lstm.csv", "output_MF.csv", "output_saint_kfold.csv"]
weight_list = [0.7385,0.6900,0.7824,0.7203,0.6330,0.7439]       # AUC 점수 기반으로 가중치 부여
df_list = [pd.read_csv(f"src/{i}")['prediction'] for i in file_list]

# %%
suffix = '_without_bert_MF'
file_list = ["output_ATTNnew.csv", "output_lgbm.csv", "output_lstm.csv", "output_saint_kfold.csv"]
weight_list = [0.7385,0.7824,0.7203,0.7439]       # AUC 점수 기반으로 가중치 부여
df_list = [pd.read_csv(f"src/{i}")['prediction'] for i in file_list]

# %%
suffix = '_softmax_without_bert_MF'
file_list = ["output_ATTNnew.csv", "output_lgbm.csv", "output_lstm.csv", "output_saint_kfold.csv"]
weight_list = [0.7385,0.7824,0.7203,0.7439]         # AUC 점수 기반으로 가중치 부여
weight_list = [5**(10*i) for i in weight_list]      # 모델들이 의견을 낼 때, (2개, 2개) 동표가 생기면 가장 성적이 좋은 lgbm의 의견이 우세하게 만듦.
weight_list = [i/max(weight_list) for i in weight_list]
df_list = [pd.read_csv(f"src/{i}")['prediction'] for i in file_list]

# %%
mean_list = [i.mean() for i in df_list]
std_list = [i.std() for i in df_list]

normalized_list = [(i-mean)/std for i, mean, std in zip(df_list, mean_list, std_list)]

# %%
N = sum(weight_list)
nonweight_prediction = sum([series for _, series in zip(weight_list, df_list)]) / len(weight_list)
avg_prediction = sum([weight * series for weight, series in zip(weight_list, df_list)]) / N

avg_norm_prediction = sum([weight * series for weight, series in zip(weight_list, normalized_list)]) / N
val_max, val_min = max(avg_norm_prediction), min(avg_norm_prediction)
gap = val_max - val_min
avg_norm_prediction = (avg_norm_prediction - val_min) / gap

# %%
avg_prediction.index.name = 'id'
avg_norm_prediction.index.name = 'id'

avg_prediction.to_csv(f'output/submission_weightedvote{suffix}.csv')
avg_norm_prediction.to_csv(f'output/submission_normalized_weightedvote{suffix}.csv')

# %%
import matplotlib.pyplot as plt

def draw_all(df_list, a=400,b=425):
    ax = plt.subplot()
    for i in df_list:
        i.iloc[a:b].plot(ax=ax)

def draw(df, a=400,b=425):
    ax = plt.subplot()
    df.iloc[a:b].plot(ax=ax)

# %%
draw_all(normalized_list)
# %%
draw(avg_norm_prediction)
draw(nonweight_prediction)
# %%
