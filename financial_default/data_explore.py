from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew
import seaborn as sns

from tryTorch.CSVDataReader import CSVDataReader
from tryTorch.DataCompressor import DataCompressor
from tryTorch.NanHandler import NanHandler
from tryTorch.OutlierHandler import FourthQuantileGapOutlierRecognizer, ThreeSigmaOutlierRecognizer
from utils.logger import logger


def get_feature_nan_statis(data: pd.DataFrame) -> pd.DataFrame:
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


def get_target_distribution(target: pd.Series) -> pd.DataFrame:
    values = set(target)
    data_dict = {value: [(target == value).sum() / len(target)] for value in values}
    print("===================target distribution===============================")
    df = pd.DataFrame(data_dict)
    for col in df.columns:
        print(f"============{col}类占比：{df.loc[0, col]}=========================")
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(map(str, df.columns)), df.loc[0, :].values)
    ax.set_title("目标值分布")
    plt.show()
    return df


def get_outlier_stat_distribution(ser: dict) -> pd.Series:
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


def get_skew_and_distribution_of_feature(features, data: pd.DataFrame, rows, cols) -> dict:
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


def get_category_feature_target_distribute(rows, cols, features, data, target, topn=10):
    """
    展示二分类型特征在不同的target下的比率
    :param rows: 画布的行数
    :param cols: 画布的列数
    :param features: 类别型特征
    :param data: 数据
    :param target: 目标变量
    :param topn: 取排名topn的子类别
    :return:
    """
    df = data[features].copy()
    target_name = None
    fig = plt.figure(figsize=(rows * 8, cols * 3))
    if isinstance(target, str):
        y = df[target]
        target_name = target
    elif isinstance(target, pd.core.series.Series) or isinstance(target, np.ndarray):
        df['y'] = target
        target_name = 'y'
    else:
        print("===========please check your target variable================")
        return
    m_rows = len(features) // cols
    for i in range(m_rows * cols):
        row = i // cols
        col = i % cols
        df[target_name] = df[target_name].astype(np.float64)
        tmp = df.groupby(features[i])[target_name].apply(np.nanmean).sort_values(ascending=False).head(topn)
        ax = plt.subplot2grid((rows, cols), (row, col), colspan=1, rowspan=1, fig=fig)
        ax.bar(tmp.index, tmp.values)
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis.set_label(features[i])
    over_num = len(features) - cols * m_rows
    if over_num == 0:
        return
    else:
        for i in range(cols * m_rows, len(features)):
            row = i // cols
            col = i % cols
            colspan = 1 if i != len(features) - 1 else cols - col
            df[target_name] = df[target_name].astype(np.float64)
            tmp = df.groupby(features[i])[target_name].apply(np.nanmean).sort_values(ascending=False).head(topn)
            ax = plt.subplot2grid((rows, cols), (row, col), colspan=colspan, rowspan=1, fig=fig)
            ax.bar(tmp.index, tmp.values)
            ax.xaxis.set_tick_params(rotation=45)
            ax.xaxis.set_label(features[i])
    fig.show()


def one_factor_va(feature: str, data: pd.DataFrame, target, alpha: float = 0.05) -> tuple:
    """
    返回单因素方差分析的P值，以及是否拒绝原假设，接受类别变量和目标变量相关的备择假设
    :param feature: 类别特征
    :param data: 数据
    :param target: 目标变量
    :param alpha: 显著性水平
    :return: （p-value, True or False）
    """
    df = data.copy()
    if isinstance(target, str):
        target_name = target
    elif isinstance(target, pd.core.series.Series) or isinstance(target, np.ndarray):
        target_name = 'y'
        df['y'] = target
    else:
        print("===========please check your target variable================")
        return
    tmp = df.groupby(by=feature).agg({target_name: ('var', 'count')})
    tmp.columns = list(map(lambda x: x[1], tmp.columns))
    SSE = tmp.assign(sse=lambda x: x['var'] * x['count']) \
        .sse.astype(np.float64).sum()
    SST = df[target_name].astype(np.float64).var() * len(df)
    SSA = SST - SSE
    k = len(df[feature].drop_duplicates())
    MSE = SSE / (len(df) - k)
    MSA = SSA / (k - 1)
    p_value = stats.f.sf(MSA / MSE, k - 1, len(df) - k)
    return p_value, p_value < alpha


def get_kde_by_target(features: list, data: pd.DataFrame, target, cols: int, rows: int):
    # 计算连续型特征在不同的目标变量下的kde
    df = data.copy()
    if isinstance(target, str):
        target_name = target
    elif isinstance(target, pd.core.series.Series) or isinstance(target, np.ndarray):
        target_name = 'y'
        df['y'] = target
    else:
        print("===========please check your target variable================")
        return
    target_values = set(df[target_name])
    fig = plt.figure(figsize=(rows * 8, cols * 3))
    for i, feature in enumerate(features):
        row, col = i // rows, i % cols
        if i < len(features) - 1:
            ax = plt.subplot2grid((rows, cols), (row, col))
        else:
            ax = plt.subplot2grid((rows, cols), (row, col), colspan=rows * cols - i)
        sns.kdeplot(df[feature], hue=df[target_name], ax=ax)
    fig.suptitle("分预测值不同特征核密度分布")
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.show()




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

    # print(get_feature_nan_statis(train_new))
    # print(get_target_distribution(Y))
    # fq = FourthQuantileGapOutlierRecognizer(float_features)
    # fq.fit_transform(train_new)
    # print(get_outlier_stat_distribution(fq.outlier_stat))
    # ts = ThreeSigmaOutlierRecognizer(float_features)
    # ts.fit_transform(train_new)
    # print(ts.outlier_stat)
    #
    # print(get_skew_and_distribution_of_feature(float_features + int_features, train_new, 7, 4))

    # print(one_factor_va('grade', train_new, Y))
    nh = NanHandler(category_features, method=3)
    nh.fit_transform(train_new)