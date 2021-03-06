# https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable


class ContractiveAutoencoder(nn.Module):
    """
    Simple contractive autoencoder with a single hidden layer.

    Constructor parameters:
        - num_inputs: Number of input features
        - num_hidden_layer_inputs: Number of input features for the single hidden layer
    """

    def __init__(self, num_inputs, num_hidden_layer_inputs):
        super(ContractiveAutoencoder, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layer_inputs = num_hidden_layer_inputs

        self.fc1 = nn.Linear(num_inputs, num_hidden_layer_inputs, bias=False)  # Encoder
        self.fc2 = nn.Linear(num_hidden_layer_inputs, num_inputs, bias=False)  # Decoder

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, self.num_inputs)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


def loss_function(W, x, recons_x, h, jacobian_weight):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `jacobian_weight` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """
    mse_loss = nn.MSELoss(size_average=False)
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h)  # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(jacobian_weight)
