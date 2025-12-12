# fnn_data.py

import numpy as np
import torch
from dataset import generate_errors, pauli_to_XZ, syndrome  # reuse his helpers 

def make_fnn_dataset(code, pX, pY, pZ, num_samples, device="cpu"):
    """
    Returns:
      X_synd : [N, n_stab] float32 tensor  (syndrome inputs)
      eX_true, eZ_true : [N, n_qubits] int arrays  (true X/Z error bits)
    """
    Hx = code.Hx.toarray()
    Hz = code.Hz.toarray()
    n_qubits = code.n

    # Generate I/X/Y/Z errors with same convention as his dataset.py
    errors = generate_errors(num_samples, n_qubits, pX, pY, pZ)  # shape [N, n_qubits]

    # Split to X/Z channels (0/1)
    eX, eZ = pauli_to_XZ(errors)  # both [N, n_qubits]

    # Compute syndromes sX, sZ as in his train_fnn.py
    sX = syndrome(Hx, eZ).T   # [N, n_Xchecks]
    sZ = syndrome(Hz, eX).T   # [N, n_Zchecks]

    s = np.concatenate([sX, sZ], axis=1)  # [N, n_stab]

    X_synd = torch.tensor(s, dtype=torch.float32, device=device)
    return X_synd, eX, eZ