import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns

from tryTorch.CSVDataReader import CSVDataReader
from tryTorch.DataCompressor import DataCompressor
from tryTorch.OutlierHandler import FourthQuantileGapOutlierRecognizer, ThreeSigmaOutlierRecognizer
from utils.logger import logger

def get_feature_nan_statis(data:pd.DataFrame)->pd.DataFrame:
    df = data.isnull().sum() / len(data)
    df = df.loc[df > 0].sort_values(ascending=False)
    contains_nan_cols_ratio = data.isnull().any().sum() / len(data.columns)
    logger.info(f"含有缺失值特征占比{contains_nan_cols_ratio}")
    print("======================nan value statis=============================")
    fig, ax = plt.subplots(1, 1)
    ax.bar(df.index, df.values)
    ax.xaxis.set_label("特征名称")
    ax.yaxis.set_label("缺失值比例")
    ax.set_title("缺失值分布")
    ax.xaxis.set_tick_params(rotation=45)
    plt.show()
    return df

def get_target_distribution(target:pd.Series)->pd.DataFrame:
    values = set(target)
    data_dict = {value: [(target==value).sum() / len(target)] for value in values}
    print("===================target distribution===============================")
    df = pd.DataFrame(data_dict)
    for col in df.columns:
        print(f"============{col}类占比：{df.loc[0,col]}=========================")
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(map(str, df.columns)), df.loc[0, :].values)
    ax.set_title("目标值分布")
    plt.show()
    return df

def get_outlier_stat_distribution(ser:dict)->pd.Series:
    ser = pd.Series(ser).sort_values(ascending=False)
    print("=====================异常值占比=======================================")
    fig, ax = plt.subplots(1, 1)
    ax.bar(ser.index, ser.values)
    ax.set_title("异常值占比分布")
    ax.xaxis.set_label("特征名称")
    ax.yaxis.set_label("异常值占比")
    ax.xaxis.set_tick_params(rotation=45)
    plt.show()
    return ser

def get_skew_and_distribution_of_feature(features, data:pd.DataFrame, rows, cols)->dict:
    """
    获取特征的核密度图，返回特征的偏度
    :param features:
    :param data:
    :param rows: 画布行数
    :param cols: 画布列数
    :return: {'feature': skew}
    """
    skews = {}
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    for i, feature in enumerate(features):
        col = i % cols
        row = i // cols
        ax = axes[row, col]
        sns.kdeplot(data[feature].astype(np.float64), ax=ax)
        skews[feature] = skew(data[feature].astype(np.float64), nan_policy='omit')
    plt.show()
    return {feature: value if isinstance(value, float) else value.data.min() for feature, value in skews.items()}


if __name__ == '__main__':
    import configparser

    # 解决matplotlib的中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    config = configparser.ConfigParser()
    config.read(r'D:\CodeProject\Python\financial_default\properties.ini')
    train_path = config["data_path"]["train_path"]
    test_path = config["data_path"]["test_path"]
    to_transform_features = config["to_transform_features"]["features"].split(",")

    data_reader = CSVDataReader(to_transform_features)
    train_data = data_reader.read(train_path)
    test_data = data_reader.read(test_path)
    Y = train_data[config["target"]["target"]].copy()

    dc = DataCompressor()
    train_data = dc.compress(train_data)
    test_data = dc.compress(test_data)

    train_new: pd.DataFrame = train_data.copy()
    test_new = train_data.copy()

    train_new.drop(columns=config["to_delete_features"]["features"].split(","), inplace=True)
    test_new = test_new[train_data.columns]

    category_features = config["category_features"]["features"].split(",")
    float_features = config["float_features"]["features"].split(",")
    int_features = config["int_features"]["features"].split(",")

    print(get_feature_nan_statis(train_new))
    print(get_target_distribution(Y))
    fq = FourthQuantileGapOutlierRecognizer(float_features)
    fq.fit_transform(train_new)
    print(get_outlier_stat_distribution(fq.outlier_stat))
    ts = ThreeSigmaOutlierRecognizer(float_features)
    ts.fit_transform(train_new)
    print(ts.outlier_stat)

    print(get_skew_and_distribution_of_feature(float_features+int_features, train_new, 7, 4))
