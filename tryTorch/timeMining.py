from datetime import datetime

import pandas as pd
from utils.logger import logger

class TimeDataMiner(object):
    #获取年份
    get_year = lambda x: x.year
    #获取季节
    get_season = lambda x: (x.month-1) // 3 + 1
    #获取月份
    get_month = lambda x: x.month
    #获取一个月的第几天
    get_day_of_month = lambda x: x.day
    #获取一个月的上中下旬
    get_day_of_order_month = lambda x: (x.day - 1) // 10 + 1
    #获取一周的第几天 %w 星期（0-6），星期天为星期的开始
    get_day_of_week = lambda x:x.strftime("%w")
    #获取小时
    get_hour_of_day = lambda x:x.hour
    #获取分钟
    get_minute_of_hour = lambda x:x.minute
    #获取秒
    get_second_of_minute = lambda x:x.second

    def __init__(self, time_feature, func:callable=lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), **kwargs):
        """
        :param time_feature: 转换为时间格式的列名
        :param func: 列=>时间列的转换函数
        """
        self.time_feature = time_feature
        self.func = func
        self.kwargs = kwargs

    def fit_transform(self, X:pd.DataFrame):
        logger.info(f"start to mine time feature: {self.time_feature}")
        x_new = X.copy()
        x_new.assign(time_col= lambda x: x[self.time_feature].apply(self.func))\
            .assign(year=lambda x:x.time_col.apply(self.get_year))\
            .assign(month=lambda x:x.time_col.apply(self.get_month))\
            .assign(season=lambda x:x.time_col.apply(self.get_season))\
            .assign(day_of_month=lambda x:x.time_col.apply(self.get_day_of_month))\
            .assign(day_of_order_month=lambda x:x.time_col.apply(self.get_day_of_order_month))\
            .assign(day_of_week=lambda x:x.time_col.apply(self.get_day_of_week))\
            .assign(hour_of_day=lambda x:x.time_col.apply(self.get_hour_of_day))\
            .assign(minute_of_hour=lambda x:x.time_col.apply(self.get_minute_of_hour))\
            .assign(second_of_minute=lambda x:x.time_col.apply(self.get_second_of_minute))
        for key, value in self.kwargs.items():
            x_new[key] = x_new.time_col.apply(value)
        logger.info(f"finished mine time feature: {self.time_feature}")
        return x_new

    def transform(self, X):
        return self.fit_transform(X)