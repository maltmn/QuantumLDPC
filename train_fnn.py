import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fnn_model import FNNDecoder
from dataset import generate_errors, pauli_to_xz, syndrome, Hx, Hz

def main():

  num_samples = 500
  n_qubits = Hx.shape[1]
  syndrome_dim = Hx.shape[0]
  pX, pY, pZ = 0.34, 0.32, 0.34

  # Generate Pauli Errors
  errors = generate_errors(num_samples, n_qubits, pX, pY, pZ)

  # Convert IXYZ to eX,eZ channels
  eX, eZ = pauli_to_XZ(errors)

  #error[sample i, qubit j]
    #eX[i,j]=1 X component
    #eZ[i,j]=1 Z component

  # Compute syndromes
  sX = syndrome(Hx, eZ).T
  sZ = syndrome(Hz, eX).T
  s = np.concatenate([sX, sZ], axis=1)

  X = torch.tensor(s, dtype=torch.float32) # syndrome input
  yX = torch.tensor(eX, dtype=torch.float32) # X errors
  yZ = torch.tensor(eZ, dtype=torch.float32) # Z errors

  # Build FNN
  model = FNNDecoder(syndrome_dim, n_qubits)

  # Compare predictions with actual error bits
  criterion = nn.BCEWithLogitsLoss() # Loss (prediction score)
  optimizer = optim.Adam(model.parameters(), lr=1e-3) # update weights (gradient descent)

  for epoch in range(20):
    optimizer.zero_grad() # reset gradients

    # Feed syndrome to output predictions
    eX_pred, eZ_pred = model(X)

    # Score Predictions
    lossX = criterion(eX_pred, yX)
    lossZ = criterion(eZ_pred, yZ)
    loss = lossX + lossZ

    # Compute how all gradient weights contributed to the error
    loss.backward() # back propagation
    optimizer.step() # step in the direction that lowers loss

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
  main()
