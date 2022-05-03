import pandas as pd

from typing import Mapping, AnyStr, List
from enum import Enum

class FeatureType(Enum):
    NUMBER = 1
    CATEGORY = 2

class ThreeSigmaOutlierRecognizer(object):
    def __init__(self, features: List):
        self.features = features
        self.feature_type = FeatureType.NUMBER
        self.stat_statis = []
        #存储异常值比率
        self.outlier_stat = {}
        self.__data_type = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def fit_transform(self, x_train:pd.DataFrame)->pd.DataFrame:
        features = set(self.features)
        features.intersection_update(set(x_train.select_dtypes(self.__data_type).columns))
        up_limit = x_train[features].mean(skipna=True) + x_train[features].std(skipna=True) * 3
        bottom_limit = x_train[features].mean(skipna=True) - x_train[features].std(skipna=True) * 3
        self.features = features
        self.stat_statis.append((bottom_limit, up_limit))
        return self.transform(x_train)

    def transform(self, x)->pd.DataFrame:
        x_new = x.copy()
        bottom = self.stat_statis[0][0]
        up = self.stat_statis[0][1]
        for feature in self.features:

            x_new = x_new.loc[(x_new[feature] >= bottom.loc[feature]) & (x_new[feature] <= up.loc[feature]), :]
        return x_new

class FourthQuantileGapOutlierRecognizer(object):
    def __init__(self, features:List[AnyStr]):
        self.features = features
        self.feature_type = FeatureType.NUMBER
        self.stat_statis = []
        self.__data_type = ['int8','int16','int32','int64', 'float16','float32','float64']

    def fit_transform(self, x_train:pd.DataFrame)->pd.DataFrame:
        features = set(self.features)
        features.intersection_update(set(x_train.select_dtypes(self.__data_type).columns))
        fourth_quantile_gap = x_train[features].quantile(0.75) - x_train[features].quantile(0.25)
        bottom_limit = x_train[features].quantile(0.25) - 1.5 * fourth_quantile_gap
        up_limit = x_train[features].quantile(0.75) + 1.5 * fourth_quantile_gap
        self.stat_statis.append((bottom_limit, up_limit))
        self.features = features
        return self.transform(x_train)

    def transform(self, x)->pd.DataFrame:
        x_new = x.copy()
        bottom = self.stat_statis[0][0]
        up = self.stat_statis[0][1]
        for feature in self.features:
            x_new = x_new.loc[(x_new[feature] >= bottom.loc[feature]) & (x_new[feature] <= up.loc[feature]), :]
        return x_new

class ObjectNumberNotEnoughOutlierRecognizer(object):
    def __init__(self, thresh: Mapping[AnyStr, int]):
        self.thresh = thresh
        self.feature_type = FeatureType.CATEGORY
        self.stat_statis = {}

    def fit_transform(self, x_train:pd.DataFrame)->pd.DataFrame:
        features = set(self.thresh.keys()).intersection(set(x_train.select_dtypes(['object'])))
        for feature in features:
            threshold = self.thresh[feature]
            cnt = x_train.groupby(feature)[feature].agg('count')
            self.stat_statis[feature] = cnt.loc[cnt >= threshold].index
        return self.transform(x_train)

    def transform(self, x)->pd.DataFrame:
        x_new = x.copy()
        for feature, values in self.stat_statis.items():
           x_new = x_new.loc[x_new[feature].isin(values), :]
        return x_new

if __name__ == '__main__':
    import numpy as np
    df = pd.DataFrame({'col1': np.random.randn(1000),'col3':np.random.randn(1000), 'col2': ['abcdeftgs'[np.random.randint(0, 9)] for _ in range(1000)]})
    ts = FourthQuantileGapOutlierRecognizer(['col1', 'col3'])
    print(ts.fit_transform(df))
    print(ts.stat_statis)
    # print(ts.transform(df))
    # sc = ObjectNumberNotEnoughOutlierRecognizer({'col2': 120})
    # print(sc.fit_transform(df))
    #
    # print(sc.transform(df))