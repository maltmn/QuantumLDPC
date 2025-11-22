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

  # Compute syndromes
  sX = syndrome(Hx, eZ)
  sZ = syndrome(Hz, eX)
  s = np.concatenate([sX, sZ] axis=1)

  X = torch.tensor(s, dtype=torch.float32)
  yX = torch.tensor(eX, dtype=torch.float32)
  yZ = torch.tensor(eZ, dtype=torch.float32)

  model = FNNDecoder(syndrome_dim, n_qubits)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(20):
    optimizer.zero_grad()

    eX_pred, eZ_pred = model(X)

    lossX = criterion(eX_pred, yX)
    lossZ = criterion(eZ_pred, yZ)
    loss = lossX + lossZ

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
  main()
