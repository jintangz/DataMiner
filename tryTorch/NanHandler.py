import pandas as pd
from enum import Enum

class AggMethod(Enum):
    MEAN = 1
    MEDIAN = 2
    MODE = 3

class NanHandler(object):
    """
    空值处理器,实现分组均值填充、中位数填充、众数填充
    """
    def __init__(self, fill_cols, category_features=None, method:int =AggMethod.MEAN):
        self.fill_cols = fill_cols
        self.category_features = category_features
        self.method = method
        self.x_stat = None

    def fit_transform(self, x_train: pd.DataFrame):
        x_new = x_train.copy()
        if self.category_features is not None:
            self.x_stat = x_new.groupby(by=self.category_features)[self.fill_cols].agg(NanHandler.group_stat, self.method)
        else:
            self.x_stat = NanHandler.group_stat(x_new, self.method)

        return self.transform(x_new)

    def transform(self, x_test):
        x_new = x_test.copy()
        #如果stat是个Series
        if isinstance(self.x_stat, pd.core.series.Series):
            return x_new[self.fill_cols].fillna(self.x_stat)
        #stat是个DataFrame
        else:
            x = self.x_stat.reset_index()
            x.index = x[self.category_features]
            x_new.index = x_new[self.category_features]
            return x_new.fillna(x).reset_index(drop=True)

    @staticmethod
    def group_stat(x: pd.DataFrame, method):
        if method == AggMethod.MEAN:
            return x.mean()
        elif method == AggMethod.MEDIAN:
            return x.median()
        elif method == AggMethod.MODE:
            return x.mode().iloc[0, :]
        else:
            return

    @staticmethod
    def my_fillna(x: pd.DataFrame, method):
        if method == AggMethod.MEAN:
            return x.fillna(x.mean())
        elif method == AggMethod.MEDIAN:
            return x.fillna(x.median())
        elif method == AggMethod.MODE:
            return x.fillna(x.mode().iloc[0, :])
        else: return

if __name__ == '__main__':
    df = pd.DataFrame({'col1':list('aaabccd'), 'col2': list('yyydddw'), 'col3':[1, 3, None, None, 2,3,3], 'col4':range(7)})
    nh = NanHandler(['col3'], method=1)
    print(nh.fit_transform(df))
    print(nh.transform(df))