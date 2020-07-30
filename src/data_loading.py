import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils import data
import numpy as np

discard_features = ['id', 'artists', 'release_date', 'name']
normalize_features = ['duration_ms', 'popularity', 'tempo', 'loudness', 'year']
batch_size = 4


def get_first_artist(artist_string):
    split_string = artist_string.replace("\"", "\'").split("\'")
    if len(split_string) < 2:
        return "unknown"
    return split_string[1]


def normalize(df, columns):
    return df.assign(**{column:
                            (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                        for column in columns})


def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)


class DataLoader:

    def __init__(self, path, popularity_cutoff=20, validation_split=0.2, shuffle=True, pca=False, pca_components=4):
        self.path = path
        self.popularity_cutoff = popularity_cutoff
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.pca = pca
        self.pca_components = pca_components
        if self.pca:
            normalize_features.extend(['PCA%i' % i for i in range(self.pca_components)])
            discard_features.remove('artists')

    def get_filtered_songs(self):
        songs = pd.read_csv(self.path)
        popular_songs = songs[songs['popularity'] >= self.popularity_cutoff]
        return popular_songs

    def get_dataset(self):
        filtered_songs = self.get_filtered_songs().drop(discard_features, axis=1)
        if self.pca:
            filtered_songs = self.artists_pca(filtered_songs)
        normalized = normalize(filtered_songs, normalize_features)
        one_hot_encoded = one_hot_encode(normalized, ['key'])
        return data.TensorDataset(torch.tensor(one_hot_encoded.values))

    def get_split(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        return indices[split:], indices[:split]

    def artists_pca(self, data):
        data['artists'] = data['artists'].astype('category')
        data['artists'] = data['artists'].apply(lambda x: get_first_artist(x))
        grouped_by_artist = data.groupby('artists').mean()
        pca = PCA(n_components=self.pca_components)
        pca_components = pd.DataFrame(pca.fit_transform(grouped_by_artist, y='artists'),
                                      columns=['PCA%i' % i for i in range(self.pca_components)],
                                      index=grouped_by_artist.index)
        merged_data = pd.merge(data, pca_components, left_on='artists', right_index=True, how='inner')
        return merged_data.drop('artists', axis=1)

    def get_preprocessed_data(self):
        dataset = self.get_dataset()
        train_indices, val_indices = self.get_split(dataset)
        train_sampler = data.SubsetRandomSampler(train_indices)
        valid_sampler = data.SubsetRandomSampler(val_indices)

        train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, validation_loader


if __name__ == '__main__':
    loader = DataLoader('data/data.csv', pca=True, pca_components=3)
    train_loader, validation_loader = loader.get_preprocessed_data()

    for X in train_loader:
        print(X[0])
