from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

data_folder = Path("data")


def plotting_outliers(df, outliers):
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False
    df['outliers'] = outliers
    # Smooth outliers (rolling window)
    _ = df['outliers'].rolling(window=12).mean().plot()
    plt.show()


def add_person_id(path):
    '''
    Obj:
    :param path: str path to data file
    :return: df with 'person' column
    '''
    df = pd.read_csv(path)
    df = df.sort_values(by=['image_id'])
    output_df = pd.DataFrame()
    for im in df.image_id.unique():
        image_df = df[df.image_id == im]
        person_list = [i for i in range(len(image_df))]
        image_df['person'] = person_list
        output_df = output_df.append(image_df)
    return output_df


def outlier_detection_ISOFOREST(df):
    df = df.fillna(0)
    clf = IsolationForest()
    outliers = clf.fit_predict(df)
    print(list(outliers).count(-1))
    return outliers


def outlier_detection_DBSCAN(df):
    df = df.fillna(0)
    outlier_detection = DBSCAN()
    outliers = outlier_detection.fit_predict(df)
    print(list(outliers).count(-1))
    return outliers


def expand_features(df, n):
    """
    Flattened n-rows vector as one row
    :param df: features df
    :param n: num rows flatten to vector
    :return:
    """
    categorical_subset = df[['image_id', 'person']]
    df.drop(columns=['image_id', 'person'], axis=1, inplace=True)
    expanded = []
    for row in df.iterrows():
        bunch = df.loc[row[0]:row[0]+n].values
        bunch_flat = bunch.flatten()
        expanded.append(bunch_flat)
    columns = [i for i in range(len(expanded[0]))]
    expanded_df = pd.DataFrame(data=expanded, columns=columns)
    df = pd.concat([expanded_df, categorical_subset], axis=1)
    return df


def log_calculation(df):
    '''
    Obj: Creates columns with log of numeric columns
    :param df: features df without ['image_id', 'person'] columns
    :return: df that contain original features and logarithm features
    '''
    categorical_subset = df[['image_id', 'person']]
    df.drop(columns=['image_id', 'person'], axis=1, inplace=True)
    numeric_subset = df.select_dtypes('number')
    for col in numeric_subset.columns:
        if col == 'image_id' or col == 'person':
            continue
        else:
            numeric_subset['log_' + col] = np.log(numeric_subset[col])
    df = pd.concat([numeric_subset, categorical_subset], axis=1)
    return df


def corr_df(df, corr_val):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''
    categorical_subset = df[['image_id', 'person']]
    df = df.drop(columns=['image_id', 'person'], axis=1)
    # Creates Correlation Matrix and Instantiates
    corr_matrix = df.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    for i in drops:
        col = df.iloc[:, (i + 1):(i + 2)].columns.values
        df = df.drop(columns=col, axis=1)

    df = pd.concat([df, categorical_subset], axis=1)
    return df


def extract_outliers(df, drop_scores=False, flatten_rows=False, log_cols=False, drop_corr=False, plotting=False):
    '''
    Obj: Finds Outliers
    :param df: features df
    :param drop_scores: bool. Drop each coordinate confidence scores.
    :param flatten_rows: bool. Create additional columns
    :param log_cols: bool. Logarithm features.
    :param drop_corr: bool. Drop high correlated features.
    :param plotting: bool. Plot results .
    :return: outliers: list
    '''

    n_expand = 10
    corr_val = 0.98


    if drop_scores:
        df.drop(columns=[str(i) for i in range(2, 50, 3, )], axis=1, inplace=True)
    if flatten_rows:
        df = expand_features(df, n_expand)
    if log_cols:
        df = log_calculation(df)
    if drop_corr:
        df = corr_df(df, corr_val)

    outliers = outlier_detection_ISOFOREST(df)
    # outliers = outlier_detection_DBSCAN(person_df)

    if plotting:
        df.index = df.image_id
        plotting_outliers(df, outliers)

    return outliers


if __name__ == '__main__':
    df = pd.read_csv(data_folder / 'Fighting042_x264.csv')

    extract_outliers(df, drop_corr=False, drop_scores=False, flatten_rows=True, log_cols=False, plotting=False)
