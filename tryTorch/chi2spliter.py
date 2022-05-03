import numpy as np
import pandas as pd

from scipy.stats import chi2


class Chi2Spliter(object):
    def __init__(self, bin_num: int = None, alpha: float = None):
        self.bin_num = bin_num
        self.alpha = alpha
        self.__split_points = None

    def fit(self, df: pd.DataFrame, x_col: str, target: str):
        cross_tab = pd.crosstab(df[x_col], df[target])
        cross_stat = cross_tab.values
        split_points = cross_tab.index.values

        limit = chi2.isf(self.alpha, df=cross_stat.shape[-1] - 1)

        while len(split_points) > self.bin_num:
            min_chi = np.inf
            min_index = -1
            for i in range(len(cross_stat) - 1):
                cal_chi2 = self.__get_chi2(cross_stat[i: i + 2])
                if cal_chi2 < min_chi:
                    min_chi = cal_chi2
                    min_index = i
            if min_chi > limit:
                break

            tmp = cross_stat[min_index] + cross_stat[min_index + 1]
            cross_stat[min_index] = tmp
            cross_stat = np.delete(cross_stat, min_index + 1, 0)
            split_points = np.delete(split_points, min_index + 1, 0)
        self.__split_points = split_points

    def transform(self, df:pd.DataFrame, col:str):
        return pd.cut(df[col], self.__split_points, right=False)

    def get_split_points(self):
        return self.__split_points

    def __get_chi2(self, arr: np.ndarray):
        row_sum = arr.sum(axis=0)
        col_sum = arr.sum(axis=1)
        N = arr.sum()
        E = np.ones(arr.shape) * row_sum.reshape(1, -1) * col_sum.reshape(-1, 1) / N
        diff = arr - E
        res = (diff * diff / E)
        res[E == 0] = 0
        return res.sum()


if __name__ == '__main__':
    target = np.random.randint(0, 2, size=1000)
    x = np.random.randn(1000)
    df = pd.DataFrame({'x':x, 'target':target})
    cs = Chi2Spliter(5, 0.5)
    cs.fit(df, 'x', 'target')
    print(cs.get_split_points())
    print(len(cs.get_split_points()))
    print(cs.transform(df, 'x'))