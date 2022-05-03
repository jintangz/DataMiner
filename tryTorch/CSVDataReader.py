import numpy as np
import pandas as pd

from utils.logger import logger

class CSVDataReader(object):
    def __init__(self, to_transform_features):
        self.features = to_transform_features

    def __transform_type(self, df):
        """
        主要是想把被错误加载为浮点型的特征转换为序数类型的特征
        :param df: 要转换的数据
        :return: 转换后的数据
        """
        logger.info("transform float features loaded in a wrong way to int...")
        logger.info(f"transforming features are as follows: {','.join(self.features)}")
        for feature in self.features:
            if feature not in df.columns:
                logger.warning(f"==================={feature} not in data=================")
                continue
            c_max = df[feature].max()
            c_min = df[feature].min()
            if c_min >= np.finfo(np.int8).min and c_max <= np.finfo(np.int8).max:
                df[feature] = df[feature].astype(np.int8)
            elif c_min >= np.finfo(np.int16).min and c_max <= np.finfo(np.int16).max:
                df[feature] = df[feature].astype(np.int16)
            elif c_min >= np.finfo(np.int32).min and c_max <= np.finfo(np.int32).max:
                df[feature] = df[feature].astype(np.int32)
            elif c_min >= np.finfo(np.int64).min and c_max <= np.finfo(np.int64).max:
                df[feature] = df[feature].astype(np.int64)
            else:
                logger.warning(f"==================={feature}: transform type failed!=============")
        return df

    def read(self, path)->pd.DataFrame:
        logger.info(f"loading data from {path}...")
        df = pd.read_csv(path)
        logger.info(f"loading data successfully...")
        return self.__transform_type(df)

