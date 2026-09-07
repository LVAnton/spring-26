from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_fscore_support
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

train = pd.read_csv("middle_train.csv")
train_y = train["label"]
train_x = train.drop('label', axis=1)

test = pd.read_csv("middle_test.csv")
test_y = test["label"]
test_x = test.drop('label', axis=1)

print("В качестве дата сета выступает случайная часть данных из набора MNIST")
print("Размер обучающей выборки: ", train_x.shape[0], "объектов")
print("Размер тестовой выборки: ", test_x.shape[0], "объектов")
print("Количество классов: ", len(train_y.unique()))

class Strategik_forest(BaseEstimator, ClassifierMixin):
    def __init__(self, n=5, max_depth=5, num_features=50, min_samples_split=20, random_state=None):
        self.n = n
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.training_data = X
        self.training_labels = y

        self.models = []
        self.features = []
        self.oob_indices = []
        self.bootstrap_indices = []

        for i in range(self.n):
            n_samples = len(X)
            bootstrap_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            oob_idx = [idx for idx in range(n_samples) if idx not in bootstrap_idx]

            self.bootstrap_indices.append(bootstrap_idx)
            self.oob_indices.append(oob_idx)

            n_features = len(X.columns)
            k = min(self.num_features, n_features)
            features_idx = self.rng.choice(n_features, size=k, replace=False)
            self.features.append(features_idx)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )

            X_bootstrap = X.iloc[bootstrap_idx].iloc[:, features_idx]
            y_bootstrap = y.iloc[bootstrap_idx]
            tree.fit(X_bootstrap, y_bootstrap)
            self.models.append(tree)

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        all_preds = []
        for i in range(self.n):
            X_selected = X.iloc[:, self.features[i]]
            pred = self.models[i].predict(X_selected)
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        final_preds = []
        for j in range(all_preds.shape[1]):
            votes = all_preds[:, j]
            most_common = np.bincount(votes).argmax()
            final_preds.append(most_common)

        return np.array(final_preds)

    def oob_score(self):
        oob_predictions = defaultdict(list)
        oob_true_labels = {}

        for tree_idx in range(self.n):
            oob_idx = self.oob_indices[tree_idx]
            if not oob_idx:
                continue

            X_oob = self.training_data.iloc[oob_idx]
            y_oob = self.training_labels.iloc[oob_idx]

            X_oob_selected = X_oob.iloc[:, self.features[tree_idx]]
            preds = self.models[tree_idx].predict(X_oob_selected)

            for position_in_oob, (idx, pred) in enumerate(zip(oob_idx, preds)):
                oob_predictions[idx].append(pred)
                if idx not in oob_true_labels:
                    oob_true_labels[idx] = y_oob.iloc[position_in_oob]

        y_true = []
        y_pred = []
        for idx in oob_true_labels:
            if oob_predictions[idx]:
                y_true.append(oob_true_labels[idx])
                votes = oob_predictions[idx]
                y_pred.append(np.bincount(votes).argmax())

        if y_true:
            return accuracy_score(y_true, y_pred)
        return 0.0

    def feature_importance_oob(self, X, y):
        base_oob = self.oob_score()
        if base_oob == 0.0:
            return {}

        importance = {}
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            oob_predictions_original = defaultdict(list)
            oob_predictions_permuted = defaultdict(list)
            oob_true_labels = {}

            for tree_idx in range(self.n):
                if feature_idx not in self.features[tree_idx]:
                    continue

                oob_idx = self.oob_indices[tree_idx]
                if not oob_idx:
                    continue

                X_oob = self.training_data.iloc[oob_idx].copy()
                y_oob = self.training_labels.iloc[oob_idx]

                pos_in_tree = np.where(self.features[tree_idx] == feature_idx)[0][0]

                X_oob_selected = X_oob.iloc[:, self.features[tree_idx]]
                preds_orig = self.models[tree_idx].predict(X_oob_selected)

                X_oob_permuted = X_oob_selected.copy()
                X_oob_permuted.iloc[:, pos_in_tree] = np.random.permutation(
                    X_oob_permuted.iloc[:, pos_in_tree].values
                )
                preds_perm = self.models[tree_idx].predict(X_oob_permuted)

                for position_in_oob, (idx, pred_orig, pred_perm) in enumerate(zip(oob_idx, preds_orig, preds_perm)):
                    oob_predictions_original[idx].append(pred_orig)
                    oob_predictions_permuted[idx].append(pred_perm)
                    if idx not in oob_true_labels:
                        oob_true_labels[idx] = y_oob.iloc[position_in_oob]

            y_true = []
            y_pred_orig = []
            y_pred_perm = []

            for idx in oob_true_labels:
                if oob_predictions_original[idx] and oob_predictions_permuted[idx]:
                    y_true.append(oob_true_labels[idx])
                    y_pred_orig.append(np.bincount(oob_predictions_original[idx]).argmax())
                    y_pred_perm.append(np.bincount(oob_predictions_permuted[idx]).argmax())

            if y_true:
                acc_orig = accuracy_score(y_true, y_pred_orig)
                acc_perm = accuracy_score(y_true, y_pred_perm)
                importance[feature_idx] = (acc_orig - acc_perm) / acc_orig * 100 if acc_orig > 0 else 0

        return importance

class MyGridSearchCV:
    def __init__(self, estimator_class, param_grid, scoring=None, verbose=1):
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.scoring = scoring or (lambda m: m.oob_score())
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    def fit(self, X, y):
        params_list = list(ParameterGrid(self.param_grid))
        for i, params in enumerate(params_list):
            model = self.estimator_class(**params, random_state=42)
            model.fit(X, y)
            score = self.scoring(model)

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                self.best_estimator_ = model

            if self.verbose:
                print(f"[{i + 1}/{len(params_list)}] {params} -> OOB={score:.4f}")
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


print("мой Random Forest")

search_my = MyGridSearchCV(
    estimator_class=Strategik_forest,
    param_grid={
        'n': [50, 100, 200],
        'max_depth': [10, 20, None],
        'num_features': [10, 28, 100, 200]
    },
    verbose=1
)

start_my = time.time()
search_my.fit(train_x, train_y)
end_my = time.time()
time_my = end_my - start_my

print("Лучшие параметры: ", search_my.best_params_)
print("Лучший OOB score: ", search_my.best_score_)
print("Время обучения: ", round(time_my, 2) )

preds_my = search_my.predict(test_x)
print("\n 10 важнейших признаков")
importance = search_my.best_estimator_.feature_importance_oob(train_x, train_y)
for j, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
    print("Признак ", j, ": ", round(imp, 2))


print("sklearn random forest")

param_grid_sklearn = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'max_features': [10, 28, 100, 200]
}

best_score_sklearn = -np.inf
best_params_sklearn = None
best_estimator_sklearn = None

params_list_sklearn = list(ParameterGrid(param_grid_sklearn))

start_sklearn = time.time()

for i, params in enumerate(params_list_sklearn):
    model = RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=-1,
        **params
    )
    model.fit(train_x, train_y)

    score = model.oob_score_

    if score > best_score_sklearn:
        best_score_sklearn = score
        best_params_sklearn = params
        best_estimator_sklearn = model

    print(f"[{i + 1}/{len(params_list_sklearn)}] {params} -> OOB={score:.4f}")

end_sklearn = time.time()
time_sklearn = end_sklearn - start_sklearn

print("Лучшие параметры: ", best_params_sklearn)
print("Лучший OOB: ", best_score_sklearn)
print("Время обучения: ", time_sklearn)


preds_sklearn = best_estimator_sklearn.predict(test_x)


print("сравнение метрик")

acc_my = accuracy_score(test_y, preds_my)
acc_sklearn = accuracy_score(test_y, preds_sklearn)

f1_macro_my = f1_score(test_y, preds_my, average='macro')
f1_macro_sklearn = f1_score(test_y, preds_sklearn, average='macro')

f1_weighted_my = f1_score(test_y, preds_my, average='weighted')
f1_weighted_sklearn = f1_score(test_y, preds_sklearn, average='weighted')

print(f""" {'Метрика':<20} {'Мой RF':<15} {'Sklearn RF':<15}
{'-' * 50}
{'Accuracy':<20} {acc_my:.4f}{'':<10} {acc_sklearn:.4f}
{'F1 (macro)':<20} {f1_macro_my:.4f}{'':<10} {f1_macro_sklearn:.4f}
{'F1 (weighted)':<20} {f1_weighted_my:.4f}{'':<10} {f1_weighted_sklearn:.4f}
{'Время (сек)':<20} {time_my:.2f}{'':<10} {time_sklearn:.2f} """)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_my = confusion_matrix(test_y, preds_my)
disp_my = ConfusionMatrixDisplay(confusion_matrix=cm_my)
disp_my.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Мой Strategik_forest', fontsize=14)

cm_sklearn = confusion_matrix(test_y, preds_sklearn)
disp_sklearn = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn)
disp_sklearn.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Sklearn RandomForest', fontsize=14)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_my_norm = cm_my.astype('float') / cm_my.sum(axis=1)[:, np.newaxis]
cm_sklearn_norm = cm_sklearn.astype('float') / cm_sklearn.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_my_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0])
axes[0].set_title('Мой RF (проценты)', fontsize=14)
axes[0].set_xlabel('Предсказано')
axes[0].set_ylabel('Реально')

sns.heatmap(cm_sklearn_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1])
axes[1].set_title('Sklearn RF (проценты)', fontsize=14)
axes[1].set_xlabel('Предсказано')
axes[1].set_ylabel('Реально')

plt.tight_layout()
plt.show()

prec_my, rec_my, f1_my, _ = precision_recall_fscore_support(test_y, preds_my, average=None)
prec_sklearn, rec_sklearn, f1_sklearn, _ = precision_recall_fscore_support(test_y, preds_sklearn, average=None)

n_classes = len(np.unique(test_y))

comparison_df = pd.DataFrame({
    'Класс': [f'Class {i}' for i in range(n_classes)],
    'Precision (Мой)': prec_my,
    'Recall (Мой)': rec_my,
    'F1 (Мой)': f1_my,
    'Precision (Sklearn)': prec_sklearn,
    'Recall (Sklearn)': rec_sklearn,
    'F1 (Sklearn)': f1_sklearn,
    'Разница F1': f1_my - f1_sklearn
})


print("Поклассовое сравнение")

print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

plt.figure(figsize=(12, 6))
x = np.arange(n_classes)
width = 0.35

plt.bar(x - width / 2, f1_my, width, label='Мой RF', color='blue', alpha=0.7)
plt.bar(x + width / 2, f1_sklearn, width, label='Sklearn RF', color='green', alpha=0.7)

plt.xlabel('Классы')
plt.ylabel('F1-score')
plt.title('Сравнение F1-score по классам')
plt.xticks(x, [f'Class {i}' for i in range(n_classes)])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
