import torch
import torch.nn as nn

#syndrome_dim = number of stabilizer rows
class FNNDecoder(nn.Module):
  def __init__(self, syndrome_dim, n_qubits):
    super().__init__()

    self.n_qubits = n_qubits
    
    self.net = nn.Sequential(

      # [5000, 15] -> [5000, 1024]
      # convert s=[0,1,0,1] to h=[3, 7.2, 0.5, -2.3...] with 1024 entries 
      nn.Linear(syndrome_dim, 1024), # input creates simple pattern

      # [5000, 1024] -> [5000, 1024]
      # keep +'s but d -'s become 0's
      nn.ReLU(),  # first hidden activation, keeps meaningful results

      # [5000, 1024] -> [5000, 1024]
      #remix h=[3, 7.2, 0.5, 0...] to h=[4.9, 5, -3, 0.1...]
      nn.Linear(1024, 1024), # hidden linear layer, creates more complex pattern

      # [5000, 1024] -> [5000, 1024]
      # keep +'s but d -'s become 0's
      nn.ReLU(), # second activation, keeps meaningful results

      # [5000, 1024] -> [5000, 32]
      nn.Linear(1024, 2*n_qubits) # output 16 X and 16 Z predictions (32 entries)
    )

  def forward(self, s):
    logits = self.net(s)
    eX_logits = logits[:, :self.n_qubits] # first 16 columns
    eZ_logits = logits[:, self.n_qubits:] # last 16 columns
    return eX_logits, eZ_logits

#runs syndrome matrix (5000 samples, 15 columns) through the neural network
#separates the 32 ouputs into eX:16 and eZ:16