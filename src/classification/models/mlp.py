# src/classification/models/mlp.py

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for classification.

    This model is dynamically constructed based on a list of
    hidden layer dimensions and a specified dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        dropout: float = 0.25,
    ):
        """
        Initializes the MLP module.

        Args:
            in_features: Dimensionality of the input features.
            hidden_dims: A list of integers, where each integer is the
                         size of a hidden layer.
            out_features: Dimensionality of the output (number of classes).
            dropout: Dropout probability to apply after each hidden layer.
        """
        super().__init__()

        layers = []
        # Track the input dimension for the current layer
        current_dim = in_features

        # Loop through the defined hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # Update the dimension for the next layer
            current_dim = h_dim

        # Add the final output layer
        # This layer outputs raw logits, as BCEWithLogitsLoss is expected
        # (as configured in 'config.yaml').
        layers.append(nn.Linear(current_dim, out_features))

        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.

        Args:
            x: The input feature tensor.
               Expected shape: (batch_size, in_features)

        Returns:
            The output logits.
            Shape: (batch_size, out_features)
        """
        return self.network(x)