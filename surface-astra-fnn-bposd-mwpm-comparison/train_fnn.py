# train_fnn.py

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from fnn_model import FNNDecoder                       # his model 
from fnn_data import make_fnn_dataset

def train_fnn_for_distance(dist, device):

    code = surface_2d.RotatedPlanar2DCode(dist)

    # use same DP channel as Astra
    error_model = PauliErrorModel(0.34, 0.32, 0.34)
    # at some training p, e.g. p_train = 0.10  (or sample a range)
    p_train = 0.10
    pi_arr, px_arr, py_arr, pz_arr = error_model.probability_distribution(code, p_train)

    # Convert to scalars (IID per qubit, as your teammate assumes)
    pX = float(px_arr[0])
    pY = float(py_arr[0])
    pZ = float(pz_arr[0])
    
    num_train = 1000
    num_test  = 100

    X_all, eX_all, eZ_all = make_fnn_dataset(code, pX, pY, pZ, num_train + num_test, device=device)
    n_qubits   = code.n
    synd_dim   = X_all.shape[1]

    X_train = X_all[:num_train]
    X_test  = X_all[num_train:]

    yX = torch.tensor(eX_all, dtype=torch.float32, device=device)
    yZ = torch.tensor(eZ_all, dtype=torch.float32, device=device)
    yX_train, yX_test = yX[:num_train], yX[num_train:]
    yZ_train, yZ_test = yZ[:num_train], yZ[num_train:]

    train_ds = TensorDataset(X_train, yX_train, yZ_train)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = FNNDecoder(synd_dim, n_qubits).to(device)

    # same weighted BCE as his script 
    pos_weight_X = (yX_train.numel() - yX_train.sum()) / yX_train.sum()
    pos_weight_Z = (yZ_train.numel() - yZ_train.sum()) / yZ_train.sum()
    pos_weight   = torch.tensor([pos_weight_X] * n_qubits, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for s_batch, x_err, z_err in train_dl:
            optimizer.zero_grad()
            eX_logits, eZ_logits = model(s_batch)
            lossX = criterion(eX_logits, x_err)
            lossZ = criterion(eZ_logits, z_err)
            loss  = lossX + lossZ
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[d={dist}] epoch {epoch+1}, loss={total_loss:.3f}")

    # simple bitwise accuracy sanity check
    model.eval()
    with torch.no_grad():
        eX_logits, eZ_logits = model(X_test)
        eX_hat = (torch.sigmoid(eX_logits) > 0.5).float()
        eZ_hat = (torch.sigmoid(eZ_logits) > 0.5).float()
        accX = (eX_hat == yX_test).float().mean().item()
        accZ = (eZ_hat == yZ_test).float().mean().item()
    print(f"[d={dist}] FNN bit accuracy: X={accX:.4f}, Z={accZ:.4f}")

    # save one checkpoint per distance
    torch.save(model.state_dict(), f"trained_models/fnn_d{dist}_best.pth")
    return model
