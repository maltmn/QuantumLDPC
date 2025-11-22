import torch
import torch.nn as nn

class FNNDecoder(nn.Module):
  def __init__(self, syndrome_dim, n_qubits):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(syndrome_dim, 256), # input
      nn.ReLU(),  # activation
      nn.Linear(256, 2*n_qubits) # output both X and Z
    )

  def forward(self, s):
    return self.net(s)
