
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
df = pd.read_csv("online_course.csv")
df = df.drop(columns=['UserID'])

for col in df.columns:
    if col != 'CourseCompletion':
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan

print("Библиотечная реализация (sklearn.tree.DecisionTreeClassifier)")

#предобработка для sklearn
def preprocess_for_sklearn(X):
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
            X_processed[col] = X_processed[col].fillna('MISSING').astype(str)
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        else:
            median_val = X_processed[col].median()
            X_processed[col] = X_processed[col].fillna(median_val)
    return X_processed

y = df['CourseCompletion']
X = df.drop(columns=['CourseCompletion'])

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_sk = preprocess_for_sklearn(X_train_sk)
X_test_sk = preprocess_for_sklearn(X_test_sk)

sk_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
sk_tree.fit(X_train_sk, y_train_sk)
pred = sk_tree.predict(X_test_sk)

print("accuracy:", accuracy_score(pred, y_test_sk))
print("f1_score:", f1_score(pred, y_test_sk, average='weighted'))
print()

import PreparingDataset
y = df['CourseCompletion']
X = df.drop(columns=['CourseCompletion'])
X['DeviceType'] = X['DeviceType'].astype(str)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp,
    test_size=0.2,
    random_state=42,
    stratify=y_tmp
)

X_train['target'] = y_train
X_val['target'] = y_val
X_test['target'] = y_test
target = 'target'
train, porogs = PreparingDataset.nums_to_cat(X_train, target)
val = PreparingDataset.nums_to_cat_test(X_val, target, porogs)
test = PreparingDataset.nums_to_cat_test(X_test, target, porogs)

real = test[target]
test = test.drop(columns=[target])

df_combined = pd.concat([train, val], ignore_index=True)

import Gain
from Gain import Node, ListNode

head = Gain.builder(train, target, Node())
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация без pruning не используя вообще валидационные данные")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real, average='weighted'))

head = Gain.builder(df_combined, target, Node())
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация без pruning, используя валидационные данные для обучения!")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real, average='weighted'))

head = Gain.builder(train, target, Node())
Gain.predict_and_fill(val, head)
Gain.pruning_tree(head, target)
preds = []
for i in range(len(test)):
    ans = Gain.get_prediction(head, test.iloc[i])
    preds.append(ans[0])
preds = [int(x) for x in preds]
print("Моя реализация с pruning")
print("accuracy:", accuracy_score(preds, real))
print("f1_score:", f1_score(preds, real, average='weighted'))
