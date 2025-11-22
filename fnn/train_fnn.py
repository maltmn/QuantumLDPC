import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fnn_model import FNNDecoder
from dataset import generate_errors, pauli_to_XZ, syndrome, Hx, Hz
from torch.utils.data import TensorDataset, DataLoader

def main():

  num_samples = 5000
  n_qubits = Hx.shape[1]
  syndrome_dim = Hx.shape[0] + Hz.shape[0]
  pX, pY, pZ = 0.02, 0.02, 0.02

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

  # Train / Test Split

  train_size = 4000
  test_size = 1000
  
  X_train = X[:train_size]
  yX_train = yX[:train_size]
  yZ_train = yZ[:train_size]

  X_test = X[train_size:train_size+test_size]
  yX_test = yX[train_size:train_size+test_size]
  yZ_test = yZ[train_size:train_size+test_size]

  train_dataset = TensorDataset(X_train, yX_train, yZ_train)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  

  # Build FNN
  model = FNNDecoder(syndrome_dim, n_qubits)

  pos_weight_X = (yX_train.numel() - yX_train.sum()) / yX_train.sum()
  pos_weight_Z = (yZ_train.numel() - yZ_train.sum()) / yZ_train.sum()

  pos_weight = torch.tensor([pos_weight_X] * n_qubits, dtype=torch.float32)

  # Compare predictions with actual error bits
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Loss (prediction score)
  optimizer = optim.Adam(model.parameters(), lr=1e-3) # update weights (gradient descent)

  epochs = 100

  for epoch in range(epochs):
    total_loss = 0
    for batchX, batchXerr, batchZerr in train_loader:

      optimizer.zero_grad() # reset gradients

      # Feed syndrome to output predictions
      eX_pred, eZ_pred = model(batchX)

      # Score Predictions
      lossX = criterion(eX_pred, batchXerr)
      lossZ = criterion(eZ_pred, batchZerr)
      loss = lossX + lossZ

      # Compute how all gradient weights contributed to the error
      loss.backward() # back propagation
      optimizer.step() # step in the direction that lowers loss

      total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

  # Evaluate

  model.eval()
  with torch.no_grad():
    eX_pred_test, eZ_pred_test = model(X_test)
    
    eX_hat = (torch.sigmoid(eX_pred_test) > 0.5).float()
    eZ_hat = (torch.sigmoid(eZ_pred_test) > 0.5).float()

    accX = (eX_hat == yX_test).float().mean().item()
    accZ = (eZ_hat == yZ_test).float().mean().item()

  print(f"Test Accuracy X: {accX:.4f}")
  print(f"Test Accuracy Z: {accZ:.4f}")

  print("\nSingle Error Decoder ")

  error_single = np.zeros((1, n_qubits), dtype=int)
  j = np.random.randint(n_qubits)

  if np.random.rand() < 0.5:
    error_single[0,j] = 1 # X
  else:
    error_single[0,j] = 2 # Z

  eX_single, eZ_single = pauli_to_XZ(error_single)

  sX_single = syndrome(Hx, eZ_single).T
  sZ_single = syndrome(Hz, eX_single).T
  s_single = np.concatenate([sX_single, sZ_single], axis=1)
  s_single_torch = torch.tensor(2*s_single - 1, dtype=torch.float32)
  with torch.no_grad():
    eX_pred_single, eZ_pred_single = model(s_single_torch)
    eX_hat_single = (torch.sigmoid(eX_pred_single) > 0.5).float()
    eZ_hat_single = (torch.sigmoid(eZ_pred_single) > 0.5).float()

  print("Pred X errors:", eX_hat_single[0].int().tolist())
  print("True X errors:", eX_single[0].tolist())

  print("\nPred Z errors:", eZ_hat_single[0].int().tolist())
  print("True Z errors:", eZ_single[0].tolist())



if __name__ == "__main__":
  main()
