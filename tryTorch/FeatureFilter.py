import pandas as pd

from utils.logger import logger

class CorrFeatureFilter(object):
    __doc__ = """
        根据特征间相关性，过滤掉高度相关的特征
    """
    def __init__(self, features, threshold:float=0.9):
        self.features = features
        self.threshold = threshold
        self.corr_matrix = None
        self.high_corr_features = None

    def fit_transform(self, x_train: pd.DataFrame):
        logger.info(f"filter features with pearson correlation index: {','.join(self.features)}")
        x_new = x_train[self.features].copy()
        corr_list = []
        for feature in self.features:
            tmp_corr = x_new.corrwith(x_new[feature], method='pearson')
            corr_list.append(tmp_corr)
        self.corr_matrix = pd.DataFrame(corr_list).rename(index={i: feature for i, feature in enumerate(self.features)})
        tmp_corr = self.corr_matrix.unstack()
        tmp_corr1 = tmp_corr.loc[(tmp_corr.abs() >= self.threshold) & (tmp_corr.abs() <= 1)]
        high_corr_feature = tmp_corr1.index.to_list()
        to_filter_feature = set()
        #保持一致性，先对每个数组中的元组进行排序
        high_corr_feature = list(map(sorted, high_corr_feature))
        self.high_corr_features = high_corr_feature
        for i, (col1, col2) in enumerate(high_corr_feature):
            if col1 == col2: continue
            to_filter_feature.add(col2)
        logger.info(f"{','.join(to_filter_feature)} will be filtered...")
        return x_new[(filter(lambda x:x not in to_filter_feature, self.features))]


if __name__ == '__main__':
    import numpy as np
    col1 = np.random.random(100)
    col2 = np.random.random(100)
    col3 = np.random.random(100) + col1
    df = pd.DataFrame({'col1': col1, 'col2': col1,'col3': col3})
    cff = CorrFeatureFilter(['col1', 'col2', 'col3'], 0.6)
    print(cff.fit_transform(df).columns)
    print(cff.corr_matrix)
    print(cff.high_corr_features)