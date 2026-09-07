import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import time

train = pd.read_csv('train_small.csv')
test = pd.read_csv('test_small.csv')

train_y = (train["label"] == 1).astype(int).values
train_x = train.drop('label', axis=1).values
test_y = (test["label"] == 1).astype(int).values
test_x = test.drop('label', axis=1).values

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_value = np.mean(y[left_mask])
                right_value = np.mean(y[right_mask])
                mse = np.sum((y[left_mask] - left_value) ** 2) + np.sum((y[right_mask] - right_value) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    self.feature_index = feature_idx
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value
        return self

    def predict(self, X):
        if self.feature_index is None:
            return np.zeros(X.shape[0])
        feature_values = X[:, self.feature_index]
        return np.where(feature_values <= self.threshold, self.left_value, self.right_value)


class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_pred = 0.0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.initial_pred = 0.0
        a = np.zeros(n_samples, dtype=np.float64)
        self.trees = []

        for t in range(self.n_estimators):
            prob = self.sigmoid(a)
            residuals = y - prob
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.trees.append(stump)
            a += self.learning_rate * stump.predict(X)
        return self

    def predict_raw(self, X):
        predictions = np.full(X.shape[0], self.initial_pred, dtype=np.float64)
        for stump in self.trees:
            predictions += self.learning_rate * stump.predict(X)
        return predictions

    def predict_proba(self, X):
        return self.sigmoid(self.predict_raw(X))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

X_train, y_train = train_x, train_y
X_test, y_test = test_x, test_y

#моя реализация
start = time.time()
gb_my = GradientBoosting(n_estimators=50, learning_rate=0.1)
gb_my.fit(X_train, y_train)
my_time = time.time() - start
my_train_acc = gb_my.score(X_train, y_train)
my_test_acc = gb_my.score(X_test, y_test)

#sklearn
start = time.time()
gb_sk = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1, random_state=42)
gb_sk.fit(X_train, y_train)
sk_time = time.time() - start
sk_train_acc = gb_sk.score(X_train, y_train)
sk_test_acc = gb_sk.score(X_test, y_test)

#кросс-валидация моя
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
my_cv = []
for train_idx, val_idx in skf.split(X_train, y_train):
    gb_cv = GradientBoosting(n_estimators=50, learning_rate=0.1)
    gb_cv.fit(X_train[train_idx], y_train[train_idx])
    my_cv.append(gb_cv.score(X_train[val_idx], y_train[val_idx]))

#кросс-валидация (sklearn)
sk_cv = cross_val_score(gb_sk, X_train, y_train, cv=5, scoring='accuracy')


print(f"{'Метрика':<20} {'Наша реализация':<20} {'sklearn':<20}")
print(f"{'Train Accuracy':<20} {my_train_acc:<20.4f} {sk_train_acc:<20.4f}")
print(f"{'Test Accuracy':<20} {my_test_acc:<20.4f} {sk_test_acc:<20.4f}")
print(f"{'CV Accuracy':<20} {np.mean(my_cv):<20.4f} {np.mean(sk_cv):<20.4f}")
print(f"{'CV Std':<20} {np.std(my_cv):<20.4f} {np.std(sk_cv):<20.4f}")
print(f"{'Time (sec)':<20} {my_time:<20.4f} {sk_time:<20.4f}")
