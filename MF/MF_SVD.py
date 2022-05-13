import numpy as np # linear algebra
import pandas as pd
import random
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, roc_auc_score

train_data = pd.read_csv('../data/train_data.csv')
test_data  = pd.read_csv('../data/test_data.csv')

# 데이터 구성
userid, itemid = list(set(train_data.userID)), list(set(train_data.assessmentItemID))
n_user, n_item = len(userid), len(itemid)

userid, itemid = list(set(test_data.userID)), list(set(test_data.assessmentItemID))
n_user, n_item = len(userid), len(itemid)

# 중복 레코드 제거
train_data.drop_duplicates(subset = ["userID", "assessmentItemID"],
                     keep = "last", inplace = True)
test_data.drop_duplicates(subset = ["userID", "assessmentItemID"],
                     keep = "last", inplace = True)

# 불필요한 column 제거
train_data.drop(['Timestamp','testId','KnowledgeTag'],
                axis=1, inplace=True, errors='ignore')

# 평가 항목 제거
test_data_old = test_data.copy()
n_user_old, n_item_old = n_user, n_item

test_data  = test_data[test_data.answerCode>=0].copy()

userid, itemid = list(set(test_data.userID)), list(set(test_data.assessmentItemID))
n_user, n_item = len(userid), len(itemid)

# 평가 항목 신규 생성
eval_data = test_data.copy()
eval_data.drop_duplicates(subset = ["userID"],
                     keep = "last", inplace = True)

# 평가 항목 테스트에서 제거
test_data.drop(index=eval_data.index, inplace=True, errors='ignore')

# pivot table
matrix_train = train_data.pivot_table('answerCode', index='userID', columns='assessmentItemID')
matrix_train.fillna(0.5, inplace=True)

# Data index mapping
user_id2idx = {v:i for i,v in enumerate(matrix_train.index)}
user_idx2id = {i:v for i,v in enumerate(matrix_train.index)}

item_id2idx = {v:i for i,v in enumerate(matrix_train.columns)}
item_idx2id = {i:v for i,v in enumerate(matrix_train.columns)}


A = matrix_train.values
a_mean = np.mean(A, axis=1)
Am = A - a_mean.reshape(-1,1)

# SVD based on matrix
U, sigma, V = svds(Am, k=12)
print(f"U={U.shape}, sigma={sigma.shape}, V={V.shape}")
print(f"Singular Vlaues : {sigma}")

# 추론을 위한 predict matrix 복원
Sigma = np.diag(sigma)
A_pred = U @ Sigma @ V + a_mean.reshape(-1,1)
restore_error = np.sum(np.square(A_pred - A)) /A_pred.size
print(f"Restore Error : {restore_error}")

# 예측 함수 정의
def predict(userid, itemid):
    useridx = user_id2idx[userid]
    itemidx = item_id2idx[itemid]
    
    return A_pred[useridx, itemidx]

a_prob = [predict(u,i) for u,i in zip(train_data.userID, train_data.assessmentItemID)]
a_pred = [round(v) for v in a_prob] 
a_true = train_data.answerCode

try:
    a_prob = [predict(u,i) for u,i in zip(test_data.userID, test_data.assessmentItemID)]
    a_pred = [round(v) for v in a_prob]
    a_true = test_data.answerCode

    print("Test data prediction")
    print(f" - Accuracy = {100*accuracy_score(a_true, a_pred):.2f}%")
    print(f" - ROC-AUC  = {100*roc_auc_score(a_true, a_prob):.2f}%")
except:
    print("Error Occurs!!")


# Test data 재현 평가
def predict(matrix, userid, itemid, user_id2idx, item_id2idx):
    
    Sigma_i = np.diag(1/sigma)
    pred_matrix = V.T @ Sigma_i @ Sigma @ V
    
    B = matrix
    B_mean = np.mean(B, axis=1)
    Bm = B - B_mean.reshape(-1,1)

    B_pred =  B @ pred_matrix + B_mean.reshape(-1,1)

    ret = [B_pred[user_id2idx[u], item_id2idx[i]] for u,i in zip(userid, itemid)]
    return ret

# 학습 데이터 재현 성공률
a_prob = predict(matrix_train.values, train_data.userID, train_data.assessmentItemID, user_id2idx, item_id2idx)
a_true = train_data.answerCode
a_pred = [round(v) for v in a_prob]

# Test data 재현 성공률
# item_id2idx는 train에서 사용한 것을 다시 사용한다.
userid = sorted(list(set([u for u in test_data.userID])))
user_id2idx_test = {v:i for i,v in enumerate(userid)}

matrix_test = 0.5*np.ones((len(userid), len(item_id2idx)))
for user,item,a in zip(test_data.userID, test_data.assessmentItemID, test_data.answerCode):
    user,item = user_id2idx_test[user],item_id2idx[item]
    matrix_test[user,item] = a

# 성능 측정
a_prob = predict(matrix_test, test_data.userID, test_data.assessmentItemID, user_id2idx_test, item_id2idx)
a_true = test_data.answerCode
a_pred = [round(v) for v in a_prob] 

print("Test data prediction")
print(f" - Accuracy = {100*accuracy_score(a_true, a_pred):.2f}%")
print(f" - ROC-AUC  = {100*roc_auc_score(a_true, a_prob):.2f}%")

# 테스트 데이터 재현 평가
a_prob = predict(matrix_test, eval_data.userID, eval_data.assessmentItemID, user_id2idx_test, item_id2idx)
a_true = eval_data.answerCode
a_pred = [round(v) for v in a_prob] 

# SVD output file 생성
import os
output_dir = "output/"
write_path = os.path.join(output_dir, "submission_MF_SVD.csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    w.write("id,prediction\n")
    for id, p in enumerate(a_pred):
        w.write("{},{}\n".format(id, p))

print("Test data prediction")
print(f" - Accuracy = {100*accuracy_score(a_true, a_pred):.2f}%")
print(f" - ROC-AUC  = {100*roc_auc_score(a_true, a_prob):.2f}%")