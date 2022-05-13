from logging import getLogger
import os
import json
import pandas as pd
import time, datetime

from recbole.model.general_recommender.lightgcn import LightGCN

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from recbole.config import Config
from recbole.data import create_dataset

from sklearn.metrics import accuracy_score, roc_auc_score

import torch

train_data = pd.read_csv('../data/train_data.csv')
test_data  = pd.read_csv('../data/test_data.csv')

data = pd.concat([train_data, test_data])

userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
n_user, n_item = len(userid), len(itemid)

data.drop_duplicates(subset = ["userID", "assessmentItemID"],
                     keep = "last", inplace = True)

data_old = data.copy()
n_user_old, n_item_old = n_user, n_item

data  = data[data.answerCode>=0].copy()

userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
n_user, n_item = len(userid), len(itemid)

eval_data = data.copy()
eval_data.drop_duplicates(subset = ["userID"],
                     keep = "last", inplace = True)

data.drop(index=eval_data.index, inplace=True, errors='ignore')


# Data File 변환
userid, itemid = sorted(list(set(data.userID))), sorted(list(set(data.assessmentItemID)))
n_user, n_item = len(userid), len(itemid)

userid_2_index = {v:i        for i,v in enumerate(userid)}
itemid_2_index = {v:i+n_user for i,v in enumerate(itemid)}
id_2_index = dict(userid_2_index, **itemid_2_index)

# 데이터 설정 파일
yamldata = """
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp

load_col:
    inter: [user_id, item_id, rating, timestamp]

user_inter_num_interval: "[0,inf)"
item_inter_num_interval: "[0,inf)"
val_interval:
    rating: "[0,1]"
    timestamp: "[97830000, inf)"
"""

# 학습데이터 저장
outpath = f"dataset/train_data"
outfile = f"dataset/train_data/train_data.inter"
yamlfile = f"train_data.yaml"

os.makedirs(outpath, exist_ok=True)

print("Processing Start")
inter_table = []
for user, item, acode, tstamp in zip(data.userID, data.assessmentItemID, data.answerCode, data.Timestamp):
    uid, iid = id_2_index[user], id_2_index[item]
    tval = int(time.mktime(datetime.datetime.strptime(tstamp, "%Y-%m-%d %H:%M:%S").timetuple()))
    inter_table.append( [uid, iid, max(acode,0), tval] )

print("Processing Complete")

print("Dump Start")
# 데이터 설정 파일 저장
with open(yamlfile, "w") as f:
    f.write(yamldata) 

# 데이터 파일 저장
with open(outfile, "w") as f:
    # write header
    f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    for row in inter_table:
        f.write("\t".join([str(x) for x in row])+"\n")

print("Dump Complete")


# 평가 데이터 저장
outpath = f"dataset/test_data"
outfile = f"dataset/test_data/test_data.inter"
yamlfile = f"test_data.yaml"

os.makedirs(outpath, exist_ok=True)

print("Processing Start")
inter_table = []
for user, item, acode, tstamp in zip(eval_data.userID, eval_data.assessmentItemID, eval_data.answerCode, eval_data.Timestamp):
    uid, iid = id_2_index[user], id_2_index[item]
    tval = int(time.mktime(datetime.datetime.strptime(tstamp, "%Y-%m-%d %H:%M:%S").timetuple()))
    inter_table.append( [uid, iid, max(acode,0), tval] )

print("Processing Complete")

print("Dump Start")
# 데이터 설정 파일 저장
with open(yamlfile, "w") as f:
    f.write(yamldata) 

# 데이터 파일 저장
with open(outfile, "w") as f:
    # write header
    f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    for row in inter_table:
        f.write("\t".join([str(x) for x in row])+"\n")

print("Dump Complete")


#Light GCN 학습
logger = getLogger()

# 설정 instance
# configurations initialization
config = Config(model='LightGCN', dataset="train_data", config_file_list=[f'train_data.yaml'])
config['epochs'] = 1
config['show_progress'] = False
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)

logger.info(config)


# Data Load
# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)


# Model instance 생성
# model loading and initialization
init_seed(config['seed'], config['reproducibility'])
model = LightGCN(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=True, show_progress=config['show_progress']
)


# 학습 결과 출력
# model evaluation
test_result = trainer.evaluate(test_data, load_best_model="True", show_progress=config['show_progress'])

logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')

result = {
    'best_valid_score': best_valid_score,
    'valid_score_bigger': config['valid_metric_bigger'],
    'best_valid_result': best_valid_result,
    'test_result': test_result
}

print(json.dumps(result, indent=4))




# 예측 및 평가 - 테스트 데이터 로드 및 평가
# configurations initialization
config = Config(model='LightGCN', dataset="test_data", config_file_list=[f'test_data.yaml'])
config['epochs'] = 1
init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)

# dataset filtering
test_dataset = create_dataset(config)
logger.info(test_dataset)

# 성능 측정
a_prob = model.predict(test_dataset).tolist()
a_true = [val for val in test_dataset.inter_feat["rating"]]
a_pred = [round(v) for v in a_prob] 

print("Test data prediction")
print(f" - Accuracy = {100*accuracy_score(a_true, a_pred):.2f}%")
print(f" - ROC-AUC  = {100*roc_auc_score(a_true, a_prob):.2f}%")

output_dir = "output/"
write_path = os.path.join(output_dir, "LightGCN(Recbole).csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    w.write("id,prediction\n")
    for id, p in enumerate(a_pred):
        w.write("{},{}\n".format(id, p))