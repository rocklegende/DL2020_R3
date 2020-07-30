import torch
import numpy

from src.architecture import ContractiveAutoencoder
from src.train import num_inputs, num_hidden_layer_inputs, save_path
from src.data_loading import DataLoader


def print_song_information(song):
    print("Song name: {}".format(song['name']))
    print("Song artists: {}".format(song['artists']))
    print("Spotify URI: spotify:track:{}".format(song['id']))
    print("Spotify URL: https://open.spotify.com/track/{}".format(song['id']))


def get_song_index(spotify_id, available_songs):
    query_result = available_songs[available_songs['id'] == spotify_id.strip()]
    query_result_row_indices = query_result.index.values.astype(int)
    if len(query_result_row_indices) > 0:
        dataframe_song_index = query_result_row_indices[0]
        # since the input dataframe could be filtered the dataframe index might not be the real index anymore
        # this is why a search for the first occurence of the dataframe index is needed to get the absolute index
        absolute_song_index = numpy.where(available_songs.index.values == dataframe_song_index)[0][0]
        return absolute_song_index
    else:
        print(
            "Could not find song with that ID, maybe it got filtered out in preprocessing or the ID doesn't exist")
        return None

def prompt_user_for_song(available_songs):
    """
    Asks the user for a Spotify song id and returns the
    row index of that song in the given DataFrame (absolute index)

    :param available_songs: songs that are available to input (pandas.DataFrame)
    :return: index of user input song (number)
    """
    absolute_song_index = None
    did_find_song_or_exited = False
    while not did_find_song_or_exited:
        input_id = input("Please input Spotify Song id: \n")
        absolute_index = get_song_index(input_id, available_songs)
        if absolute_index is not None:
          return absolute_index

    # return absolute_song_index


def get_song_recommendations(input_song_index, k, loader):
    """
    :param input_song_index: Index of the song to search similar songs for
    :param k: Top k nearest songs
    :param loader: DataLoader
    :return: list of DataFrame rows (pandas.core.series.Series)
    """
    # load model parameters from disk and initialize model with it
    trained_model = ContractiveAutoencoder(num_inputs, num_hidden_layer_inputs).double()
    trained_model.load_state_dict(torch.load(save_path))
    trained_model.eval()

    song_dataset = loader.get_dataset()
    all_songs = loader.get_filtered_songs()

    print("Found song: ")
    input_song = all_songs.iloc[input_song_index]
    print_song_information(input_song)

    print('')
    print('Searching for top {} similar songs..'.format(k))

    """
    Compute dense song representation for every song of the data set and create tensor from it
    """
    all_songs_dense_list = []
    for song in song_dataset:
        dense_song_representation, song_reconstruction = trained_model(song[0])
        all_songs_dense_list.append(dense_song_representation)
    all_songs_dense = torch.cat(all_songs_dense_list)

    input_song_dense = all_songs_dense[input_song_index]

    """
    Drop the input song from the dense representations and meta lookup, so we don't compare input with itself
    """
    all_songs_without_input_song = all_songs.drop(all_songs.index[input_song_index])
    all_songs_dense_without_input_song = torch.cat(
        (all_songs_dense[0:input_song_index], all_songs_dense[input_song_index + 1:]))

    """
    Search top k nearest songs based on the euclidean distance
    between the input dense representation and all other representations.
    """
    euclidean_distances = torch.norm(all_songs_dense_without_input_song - input_song_dense, dim=1, p=None)
    # k = 10
    topk_nearest_songs = euclidean_distances.topk(k, largest=False)

    recommendations = []
    for index_tensor in topk_nearest_songs.indices:
        index = index_tensor.item()
        recommended_song = all_songs_without_input_song.iloc[index]
        recommendations.append(recommended_song)

    return recommendations


def print_recommendations(recommendations):
    print('')
    print("======RESULT======")
    rank = 1
    for song in recommendations:
        print(rank)
        print_song_information(song)
        print("==================")
        rank += 1


if __name__ == '__main__':
    loader = DataLoader('data/data.csv')
    input_song_index = prompt_user_for_song(loader.get_filtered_songs())
    recommendations = get_song_recommendations(input_song_index, 10, loader)
    print_recommendations(recommendations)