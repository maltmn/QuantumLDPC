import torch
import torch.nn as nn

class FNNDecoder(nn.Module):
  def __init__(self, syndrome_dim, n_qubits):
    super().__init__()

    self.n_qubits = n_qubits
    
    self.net = nn.Sequential(
      nn.Linear(syndrome_dim, 1024), # input
      nn.ReLU(),  # activation
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 2*n_qubits) # output both X and Z
    )

  def forward(self, s):
    logits = self.net(s)
    eX_logits = logits[:, :self.n_qubits]
    eZ_logits = logits[:, self.n_qubits:]
    return eX_logits, eZ_logits
