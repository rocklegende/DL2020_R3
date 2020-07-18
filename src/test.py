import torch

from src.architecture import ContractiveAutoencoder
from src.train import num_inputs, num_hidden_layer_inputs, save_path
from src.data_loading import DataLoader

# load model from disk
trained_model = ContractiveAutoencoder(num_inputs, num_hidden_layer_inputs)
trained_model.load_state_dict(torch.load(save_path))
trained_model.eval()

# now do something with the loaded trained model..
# display model output just for first song of the validation set
loader = DataLoader('data/data.csv')
train_loader, validation_loader = loader.get_preprocessed_data()
count = 0
for data in validation_loader:
    if count < 1:
        song = data[0][0].double()
        trained_model = trained_model.double()
        dense_song_representation, song_reconstruction = trained_model(song)
        print("Original input features of song:")
        print(song)
        print('================')
        print("Dense representation of song:")
        print(dense_song_representation)
        print('================')
        print("Reconstructed song:")
        print(song_reconstruction)
        print('================')
        count += 1
