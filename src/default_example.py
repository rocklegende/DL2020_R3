from src.test import get_song_recommendations, print_recommendations
from src.data_loading import DataLoader


if __name__ == '__main__':
    loader = DataLoader('data/data.csv')
    default_song_index = 0
    recommendations = get_song_recommendations(default_song_index, 10, loader)
    print_recommendations(recommendations)
