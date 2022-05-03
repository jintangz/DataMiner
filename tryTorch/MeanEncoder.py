from itertools import product
from typing import List, AnyStr

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


class MeanEncoder(object):
    def __init__(self, category_features: List[AnyStr], n_splits: int = 5, task_type='classification',
                 prior_weight_func=None, random_state=0):
        self.category_features = category_features
        self.n_splits = n_splits
        self.task_type = task_type
        # 存储根据交叉验证学习的先验和计算的编码值
        self.learned_stats = {}

        if callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        elif isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

        if task_type != 'classification':
            self.task_type = 'regression'
            self.target_values = None
        else:
            self.task_type = task_type
            self.target_values = []

        self.sk = StratifiedKFold(self.n_splits, random_state=random_state) if task_type == 'classification' else KFold(
            self.n_splits, random_state=random_state)

    def fit_transform(self, X, y):
        x_temp = X.copy()
        if self.task_type == 'classification':
            self.targets = set(y)
            self.learned_stats = {f"{feature}_pred_{target}": [] for feature, target in
                                  product(self.category_features, self.targets)}
            for feature, target in product(self.category_features, self.targets):
                nf_name = "{}_pred_{}".format(feature, target)
                x_temp.loc[:, nf_name] = None
                for large_index, small_index in self.sk.split(X, y):
                    nf_train, nf_test, prior, col_avg_y = self.__encode(feature, target, x_temp.iloc[large_index],
                                                                        x_temp.iloc[small_index], y[large_index])
                    x_temp.loc[small_index, nf_name] = nf_test
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {f"{feature}_pred": [] for feature in self.category_features}
            for feature in self.category_features:
                nf_name = f"{feature}_pred"
                x_temp[:, nf_name] = None
                for large_index, small_index in self.sk.split(X, y):
                    nf_train, nf_test, prior, col_avg_y = self.__encode(feature, None, x_temp.iloc[large_index],
                                                                        x_temp.iloc[small_index], y[large_index])
                    x_temp.loc[small_index, nf_name] = nf_test
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return x_temp

    def transform(self, X_test):
        X = X_test.copy()
        if self.task_type == 'classification':
            for feature, target in product(self.category_features, self.targets):
                nf_name = f"{feature}_pred_{target}"
                X[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X[nf_name] += X[[feature]].merge(col_avg_y, how='left', on=feature). \
                        fillna(prior, inplace=False)[nf_name]
                X[nf_name] /= self.n_splits
        else:
            for feature in self.category_features:
                nf_name = f"{feature}_pred"
                X[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X[nf_name] += X[[feature]].merge(col_avg_y, how='left', on=feature). \
                        fillna(prior, inplace=False)[nf_name]
                X[nf_name] /= self.n_splits
        return X

    def __encode(self, feature, target, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train):
        x_train_new = x_train[[feature]].copy()
        x_test_new = x_test[[feature]].copy()
        if self.task_type == 'classification':
            nf_name = "{}_pred_{}".format(feature, target)
            x_train_new['pre_temp'] = (y_train == target).astype(int)
        else:
            nf_name = f"{feature}_pred"
            x_train_new['pre_temp'] = y_train

        # 计算先验概率/均值
        prior = x_train_new['pre_temp'].mean()
        # 计算后验概率/均值和数量
        y_tmp = x_train_new.groupby(by=feature).agg({'pre_temp': ['mean', 'count']})
        y_tmp.columns = list(map(lambda x: x[1], y_tmp.columns))
        # 计算权重
        y_tmp['beta'] = self.prior_weight_func(y_tmp['count'])
        # 计算加权的编码概率/均值
        y_tmp[nf_name] = y_tmp['beta'] * prior + (1 - y_tmp['beta']) * y_tmp['mean']
        y_tmp.drop(axis=1, columns=['count', 'mean']).reset_index(inplace=True)

        nf_train = x_train_new.merge(y_tmp, how='left', on=feature)[nf_name].values
        nf_test = x_test_new.merge(y_tmp, how='left', on=feature)[nf_name].fillna(prior, inplace=False).values

        return nf_train, nf_test, prior, y_tmp


if __name__ == '__main__':
    df = pd.DataFrame({'x1': np.random.randint(1, 20, 100), 'x2': ['abc'[np.random.randint(0, 3)] for _ in range(100)],
                       'y': np.random.randint(0, 2, 100)})

    me = MeanEncoder(['x1'], 3)
    print(me.fit_transform(df[['x1']], df['y']))

    print(me.transform(df[['x1']]))
