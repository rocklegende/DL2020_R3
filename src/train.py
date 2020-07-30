import torch
import torch.optim as optim
from src.architecture import ContractiveAutoencoder, loss_function
from src.data_loading import DataLoader


class ModelTrainer:
    r"""
        Model trainer. Performs the training on the model by calling ModelTrainer.train()

        Arguments:
            model: The model to train
            loader: The data loader
            num_inputs (int): Number of input features.
            jacobian_weight (float): the weight given to the jacobian regulariser term.
            num_epochs (int): Number of epochs to perform on training
            learning_rate (float): The learning rate to apply
        """

    def __init__(self, model, loader, num_inputs, jacobian_weight=1e-4, num_epochs=15, learning_rate=0.0001):
        self.model = model
        self.num_inputs = num_inputs
        self.jacobian_weight = jacobian_weight
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.loader = loader
        train_loader, validation_loader = loader.get_preprocessed_data()
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.model = self.model.double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_model_weights(self):
        return self.model.state_dict()['fc1.weight']

    def train(self):
        for epoch in range(self.num_epochs):
            count = 0
            self.model.train()
            train_loss = 0
            for data in self.train_loader:
                self.optimizer.zero_grad()
                data_as_tensor = data[0].double()
                hidden_representation, reconstruction_x = self.model(data_as_tensor)

                weights = self.get_model_weights()
                loss = loss_function(weights, data_as_tensor.view(-1, self.num_inputs), reconstruction_x,
                                     hidden_representation, self.jacobian_weight)

                loss.backward()
                train_loss += loss.data[0]
                self.optimizer.step()
                count += 1

            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, train_loss / len(self.train_loader.dataset)))

    def save_model(self, path='model.pt'):
        torch.save(self.model.state_dict(), path)


num_inputs = 26
num_hidden_layer_inputs = 10
jacobian_weight = 1e-4
num_epochs = 10
learning_rate = 0.0001
save_path = 'model.pt'

if __name__ == '__main__':
    autoencoder_model = ContractiveAutoencoder(num_inputs, num_hidden_layer_inputs)
    loader = DataLoader('data/data.csv')
    # instantiate model trainer
    model_trainer = ModelTrainer(autoencoder_model, loader, num_inputs, jacobian_weight, num_epochs, learning_rate)
    # perform training
    model_trainer.train()
    # save model
    model_trainer.save_model(save_path)
