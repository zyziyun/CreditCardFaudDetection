import pandas as pd
from torch.utils.data import Dataset, random_split


class DLoader(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, header=None, skiprows=1)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, id):
        return [self.X[id], self.y[id]]

    def test_split(self, nums=0.33):
        test_size = round(nums * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

train_df, test_df = DLoader('../creditcard.csv').test_split(0.33)
print(train_df, test_df)