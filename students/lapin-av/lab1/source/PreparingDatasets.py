import pandas as pd

def num_to_cat(df, col_name):
    "Функция допускает ошибки, если количество групп примерно, как количество значений или меньше."
    "Но для большого дата сета все нормально работает"
    none_num = df[col_name].isna().sum()
    df[col_name] = df[col_name].apply(str)
    col = df[col_name]


    num_groups = 3
    col = col.sort_values()
    group_names = ['a', 'b', 'c']
    group_size = (len(col) - none_num) // num_groups + 1

    porog_info = []
    for i in range(1, num_groups):
        #print(col)
        edge = (float(col.iloc[group_size * i + 1]) + float(col.iloc[group_size * i])) / 2
        porog_info.append(edge)

    for i in range(len(col)):
        if col.iloc[i] != 'nan':
            group_name = group_names[i // group_size]
            col.iloc[i] = group_name
    col = col.sort_index()
    df[col_name] = col
    return df, porog_info, group_names

def num_to_cat_test(df, col, porogs, letters):
    df[col] = df[col].apply(str)
    for i in range(len(df)):
        if df[col].iloc[i] != 'nan':
            num = float(df[col].iloc[i])
            if num <= porogs[0]:
                df.loc[df.index[i], col] = letters[0]
            elif num > porogs[-1]:
                df.loc[df.index[i], col] = letters[-1]
            else:
                for j in range(len(porogs)):
                    if num > porogs[j] and num <= porogs[j + 1]:
                        df.loc[df.index[i], col] = letters[j + 1]
    return df

def nums_to_cat(df, target):
    y = df[target]
    df = df.drop(columns=[target])
    columns = df.columns
    dicter = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df, porog_info, letters = num_to_cat(df, col)
            dicter[col] = (porog_info, letters)
    df[target] = y
    df['w'] = 1.0
    return df, dicter

def nums_to_cat_test(df, target, porog_info):
    y = df[target]
    df = df.drop(columns=[target])
    columns = df.columns
    dicter = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            porogs = porog_info[col][0]
            letters = porog_info[col][1]
            df = num_to_cat_test(df, col, porogs, letters)
    df[target] = y
    df['w'] = 1.0
    return df

def split_dataset(df):
    proportions = (0.6, 0.2, 0.2)
    random_state = 42
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(shuffled_df)
    first_idx = int(n * proportions[0])
    second_idx = first_idx + int(n * proportions[1])
    df1 = shuffled_df.iloc[:first_idx].copy()
    df2 = shuffled_df.iloc[first_idx:second_idx].copy()
    df3 = shuffled_df.iloc[second_idx:].copy()
    return df1, df2, df3

def prepare_dataset(df, target):
    train, val, test = split_dataset(df)
    train, porogs = nums_to_cat(train, target)
    val = nums_to_cat_test(val, target, porogs)
    test = nums_to_cat_test(test, target, porogs)
    return train, val, test
