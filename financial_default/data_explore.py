import pandas as pd

from tryTorch.CSVDataReader import CSVDataReader
from tryTorch.DataCompression import DataCompressor

if __name__ == '__main__':
    import configparser
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

    train_new:pd.DataFrame = train_data.copy()
    test_new = train_data.copy()

    train_new.drop(columns=config["to_delete_features"]["features"].split(","), inplace=True)
    test_new = test_new[train_data.columns]

