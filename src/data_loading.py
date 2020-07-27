import pandas as pd
import torch
from torch.utils import data
import numpy as np

discard_features = ['id', 'artists', 'release_date', 'name']
normalize_features = ['duration_ms', 'popularity', 'tempo', 'loudness', 'year']
batch_size = 10


class DataLoader:

    def __init__(self, path, popularity_cutoff=20, validation_split=0.2, shuffle=True):
        self.path = path
        self.popularity_cutoff = popularity_cutoff
        self.validation_split = validation_split
        self.shuffle = shuffle

    def normalize(self, df, columns):
        return df.assign(**{column:
                            (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                            for column in columns})

    def one_hot_encode(self, df, columns):
        return pd.get_dummies(df, columns=columns)


    def get_filtered_songs(self):
        songs = pd.read_csv(self.path)
        popular_songs = songs[songs['popularity'] >= self.popularity_cutoff]
        return popular_songs

    def get_dataset(self):
        filtered_songs = self.get_filtered_songs().drop(discard_features, axis=1)
        normalized = self.normalize(filtered_songs, normalize_features)
        one_hot_encoded = self.one_hot_encode(normalized, ['key'])
        return data.TensorDataset(torch.tensor(one_hot_encoded.values))

    def get_split(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        return indices[split:], indices[:split]

    def get_preprocessed_data(self):
        dataset = self.get_dataset()
        train_indices, val_indices = self.get_split(dataset)
        train_sampler = data.SubsetRandomSampler(train_indices)
        valid_sampler = data.SubsetRandomSampler(val_indices)

        train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, validation_loader


if __name__ == '__main__':
    loader = DataLoader('data/data.csv')
    train_loader, validation_loader = loader.get_preprocessed_data()

    for X in train_loader:
        print(X[0])
