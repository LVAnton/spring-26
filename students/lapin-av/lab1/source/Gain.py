import pandas as pd
import numpy as np
import PreparingDataset

# Гиперпараметр
min_gain_lvl = 0.01


def calculate_ginni(dicter):
    N = sum(dicter.values())
    summ = 0
    for val in dicter.values():
        summ += (val / N) ** 2
    return 1 - summ, N


def count_gain(parent_dicter, dicters):
    parent_ginni, N = calculate_ginni(parent_dicter)
    child_ginni = 0
    for dict in dicters.values():
        ginni, n = calculate_ginni(dict)
        child_ginni += ginni * (n / N)
    gain = parent_ginni - child_ginni
    return gain


def form_dicters(part, col_name, target):
    return {
        category: group.groupby(target)['w'].sum().to_dict()
        for category, group in part.groupby(col_name)
    }


def form_parent_dicter(part, col_name, target):
    return part.groupby(target)['w'].sum().to_dict()


def gain(part, col_name, target):
    dicters = form_dicters(part, col_name, target)
    dicter = form_parent_dicter(part, col_name, target)
    gain = count_gain(dicter, dicters)
    return gain


# Новая функция для бинарного разбиения числовых признаков
def find_best_binary_threshold(df, feature, target):
    """Находит оптимальный порог для бинарного разбиения числового признака"""
    non_null = df[df[feature].notna()]
    if len(non_null) < 2:
        return None, -1

    sorted_df = non_null.sort_values(feature)
    unique_values = sorted_df[feature].unique()

    best_gain = -1
    best_threshold = None

    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        left = sorted_df[sorted_df[feature] <= threshold]
        right = sorted_df[sorted_df[feature] > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        # Используем существующую функцию gain
        gain_value = gain_for_binary_split(df, feature, threshold, target)

        if gain_value > best_gain:
            best_gain = gain_value
            best_threshold = threshold

    return best_threshold, best_gain


def gain_for_binary_split(part, feature, threshold, target):
    """Вычисляет прирост информации для бинарного разбиения"""
    left_part = part[part[feature] <= threshold]
    right_part = part[part[feature] > threshold]

    left_dict = left_part.groupby(target)['w'].sum().to_dict()
    right_dict = right_part.groupby(target)['w'].sum().to_dict()
    parent_dict = part.groupby(target)['w'].sum().to_dict()

    parent_gini, N = calculate_ginni(parent_dict)
    left_gini, n_left = calculate_ginni(left_dict)
    right_gini, n_right = calculate_ginni(right_dict)

    child_gini = left_gini * (n_left / N) + right_gini * (n_right / N)
    return parent_gini - child_gini


# Новая функция для бинарного разбиения категориальных признаков
def find_best_categorical_split(df, feature, target):
    """Находит лучшее бинарное разбиение категориального признака"""
    categories = df[feature].unique()
    if len(categories) < 2:
        return None, -1

    best_gain = -1
    best_split = None

    for cat in categories:
        left = df[df[feature] == cat]
        right = df[df[feature] != cat]

        if len(left) == 0 or len(right) == 0:
            continue

        left_dict = left.groupby(target)['w'].sum().to_dict()
        right_dict = right.groupby(target)['w'].sum().to_dict()
        parent_dict = df.groupby(target)['w'].sum().to_dict()

        parent_gini, N = calculate_ginni(parent_dict)
        left_gini, n_left = calculate_ginni(left_dict)
        right_gini, n_right = calculate_ginni(right_dict)

        child_gini = left_gini * (n_left / N) + right_gini * (n_right / N)
        gain_value = parent_gini - child_gini

        if gain_value > best_gain:
            best_gain = gain_value
            best_split = [cat]

    return best_split, best_gain


def choose_best_feature(df, target):
    cols = df.columns.to_list()
    cols.pop()
    cols.pop()
    if len(cols) == 0:
        return None, None, False

    max_gain = 0
    best_col = 0
    best_threshold = None
    is_numeric = False

    for col in cols:
        # Проверяем тип признака
        if pd.api.types.is_numeric_dtype(df[col]):
            # Числовой признак - ищем оптимальный порог
            threshold, new_gain = find_best_binary_threshold(df, col, target)
            if threshold is not None and new_gain > max_gain:
                max_gain = new_gain
                best_col = col
                best_threshold = threshold
                is_numeric = True
        else:
            # Категориальный признак - ищем лучшее бинарное разбиение
            split, new_gain = find_best_categorical_split(df, col, target)
            if split is not None and new_gain > max_gain:
                max_gain = new_gain
                best_col = col
                best_threshold = split
                is_numeric = False

    if max_gain < min_gain_lvl:
        return None, None, False
    return best_col, best_threshold, is_numeric


def feature_decomposition(df, col_name, threshold, is_numeric):
    datasets = []
    nan_rows = []
    dicter = {}
    col = df[col_name]
    df = df.drop(columns=[col_name])

    # Разбиваем на две группы
    if is_numeric:
        for i in range(len(col)):
            row = df.iloc[i]
            group = col.iloc[i]
            if group is None or group == 'nan' or pd.isna(group):
                nan_rows.append(row)
            else:
                if group <= threshold:
                    if 0 in dicter:
                        datasets[dicter[0]].append(row.to_dict())
                    else:
                        dicter[0] = len(dicter)
                        datasets.append([row.to_dict()])
                else:
                    if 1 in dicter:
                        datasets[dicter[1]].append(row.to_dict())
                    else:
                        dicter[1] = len(dicter)
                        datasets.append([row.to_dict()])
    else:
        # Категориальный признак
        for i in range(len(col)):
            row = df.iloc[i]
            group = col.iloc[i]
            if group is None or group == 'nan' or pd.isna(group):
                nan_rows.append(row)
            else:
                if group in threshold:  # left ветвь
                    if 0 in dicter:
                        datasets[dicter[0]].append(row.to_dict())
                    else:
                        dicter[0] = len(dicter)
                        datasets.append([row.to_dict()])
                else:  # right ветвь
                    if 1 in dicter:
                        datasets[dicter[1]].append(row.to_dict())
                    else:
                        dicter[1] = len(dicter)
                        datasets.append([row.to_dict()])

    weights = []
    for i in range(len(datasets)):
        summ = 0
        for j in range(len(datasets[i])):
            summ += datasets[i][j]['w']
        weights.append(summ)
    N = sum(weights)
    probs = []
    for elem in weights:
        if N > 0:
            probs.append(elem / N)
        else:
            probs.append(0.5)

    # Обработка пропусков
    for i in range(len(nan_rows)):
        raw = nan_rows[i]
        for j in range(len(probs)):
            row = raw.copy()
            row['w'] = raw['w'] * probs[j]
            datasets[j].append(row.to_dict())

    for i in range(len(datasets)):
        datasets[i] = pd.DataFrame(datasets[i])

    dicter2 = {}  # для сохранения вероятностей
    for elem in dicter.keys():
        dicter2[elem] = probs[dicter[elem]]
        dicter[elem] = datasets[dicter[elem]]

    return dicter, dicter2


def decompose_dataset(df, target):
    feature, threshold, is_numeric = choose_best_feature(df, target)
    if feature is None:
        return None, None, None, None, None
    data, probs = feature_decomposition(df, feature, threshold, is_numeric)
    return data, feature, probs, threshold, is_numeric


class Node:
    def __init__(self):
        self.feature = None  # признак, который будет правилом
        self.threshold = None  # порог для бинарного разбиения
        self.is_numeric = None  # является ли признак числовым
        self.link = None  # На родителя
        self.kids = {}  # Пустрой словарь детей (теперь всегда 2)
        self.probs = {}  # Пустой словарь вероятностей
        self.ans = None  # Потенциальный ответ, для pruning сохраняем
        self.data = None  # Только при pruning будет заполено данными
        self.correct = None  # Для pruning


class ListNode:
    def __init__(self):
        self.ans = None  # Класс, который мы берем
        self.link = None  # На родителя
        self.data = None  # На этапе построения не нужна. Для pruning нужно!
        self.correct = None  # Для pruning


# Придумать, как строить и хранить дерево, а так же придумать, как я буду это предсказывать!
def builder(df, target, parent_node):
    data, feature, probs, threshold, is_numeric = decompose_dataset(df, target)
    if data is None:  # Лист по причине отсутствия признаков для разбиения или слаботы прироста информативности
        list_node = ListNode()
        list_node.link = parent_node
        list_node.ans = df[target].mode()[0]
        return list_node
    else:
        curr_node = Node()
        curr_node.link = parent_node
        curr_node.feature = feature
        curr_node.threshold = threshold
        curr_node.is_numeric = is_numeric
        curr_node.probs = probs
        curr_node.ans = df[target].mode()[0]
        for key in data.keys():
            curr_node.kids[key] = builder(data[key], target, curr_node)
        return curr_node


def predict_and_fill(df, now):
    if type(now) is Node:
        now.data = df

        # Проверяем, есть ли данные для разбиения
        if len(df) == 0:
            return

        for i in range(len(df)):
            if pd.isna(df[now.feature].iloc[i]):
                df.loc[i, now.feature] = None

        # Разбиваем на две ветви
        if now.is_numeric:
            df_not_none = df[df[now.feature].notna()]
            df_none = df[df[now.feature].isna()]

            left_mask = df_not_none[now.feature] <= now.threshold
            right_mask = df_not_none[now.feature] > now.threshold

            left_data = df_not_none[left_mask]
            right_data = df_not_none[right_mask]
        else:
            df_not_none = df[df[now.feature].notna()]
            df_none = df[df[now.feature].isna()]

            left_mask = df_not_none[now.feature].isin(now.threshold)
            right_mask = ~df_not_none[now.feature].isin(now.threshold)

            left_data = df_not_none[left_mask]
            right_data = df_not_none[right_mask]

        # Обработка пропусков
        if len(df_none) > 0:
            total_non_nan = len(left_data) + len(right_data)
            if total_non_nan > 0:
                left_prob = len(left_data) / total_non_nan
                right_prob = len(right_data) / total_non_nan
            else:
                left_prob = 0.5
                right_prob = 0.5

            left_none = df_none.copy()
            right_none = df_none.copy()
            left_none['w'] = df_none['w'] * left_prob
            right_none['w'] = df_none['w'] * right_prob

            left_data = pd.concat([left_data, left_none], ignore_index=True)
            right_data = pd.concat([right_data, right_none], ignore_index=True)

        if len(left_data) > 0:
            left_data = left_data.drop(columns=[now.feature])
        if len(right_data) > 0:
            right_data = right_data.drop(columns=[now.feature])

        # Проверяем, что ключи совпадают с детьми
        if 0 in now.kids and len(left_data) > 0:
            predict_and_fill(left_data, now.kids[0])
        if 1 in now.kids and len(right_data) > 0:
            predict_and_fill(right_data, now.kids[1])

    else:
        now.data = df


def pruning_tree(now, target):
    if type(now) == ListNode:
        now.correct = 0
        # Проверяем, что data существует и не пуста
        if now.data is not None and len(now.data) > 0:
            for i in range(len(now.data)):
                if now.data.iloc[i][target] == now.ans:
                    now.correct += now.data.iloc[i]['w']
        return now
    else:
        # Рекурсивно применяем pruning к детям
        for key in list(now.kids.keys()):
            now.kids[key] = pruning_tree(now.kids[key], target)

        my_count_correct = 0
        # Проверяем, что data существует и не пуста
        if now.data is not None and len(now.data) > 0:
            df = now.data
            for i in range(len(df)):
                if df.iloc[i][target] == now.ans:
                    my_count_correct += df.iloc[i]['w']
        else:
            # Если данных нет, считаем, что лучше оставить как есть
            now.correct = 0
            for key in now.kids.keys():
                now.correct += now.kids[key].correct
            return now

        correct_nothing_summ = 0
        for key in now.kids.keys():
            correct_nothing_summ += now.kids[key].correct

        if my_count_correct >= correct_nothing_summ:
            new_list = ListNode()
            new_list.link = now.link
            new_list.data = now.data
            new_list.ans = now.ans
            new_list.correct = my_count_correct
            return new_list
        else:
            now.correct = correct_nothing_summ
            return now


def get_prediction(now, row):
    if type(now) == ListNode:
        return now.ans, row['w']
    else:
        # Обработка пропусков
        if pd.isna(row[now.feature]) or row[now.feature] is None:
            results = {}
            total_weight = 0

            for key, prob in now.probs.items():
                new_row = row.copy()
                new_row['w'] = row['w'] * prob

                ans, weight = get_prediction(now.kids[key], new_row)

                if ans in results:
                    results[ans] += weight
                else:
                    results[ans] = weight
                total_weight += weight

            if results:
                best_ans = max(results.items(), key=lambda x: x[1])[0]
                return best_ans, total_weight
            else:
                return now.ans, row['w']

        # Нет пропуска
        if now.is_numeric:
            if row[now.feature] <= now.threshold:
                if 0 in now.kids:
                    return get_prediction(now.kids[0], row)
                else:
                    return now.ans, row['w']
            else:
                if 1 in now.kids:
                    return get_prediction(now.kids[1], row)
                else:
                    return now.ans, row['w']
        else:
            if row[now.feature] in now.threshold:
                if 0 in now.kids:
                    return get_prediction(now.kids[0], row)
                else:
                    return now.ans, row['w']
            else:
                if 1 in now.kids:
                    return get_prediction(now.kids[1], row)
                else:
                    return now.ans, row['w']
