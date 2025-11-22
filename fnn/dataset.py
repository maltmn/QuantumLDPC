import numpy as np

def generate_errors(num_samples, n_qubits, pX, pY, pZ):
# PauliErrorModel(0.34, 0.32, 0.34)
  errors = np.random.choice(
    [0,1,2,3], # I,X,Y,Z
    size = (num_samples, n_qubits),
    p = [1-pX-pY-pZ, pX, pY, pZ]
  )
  return errors

def pauli_to_XZ(errors):
  eX = ((errors == 1) | (errors == 2)).astype(int) # X or Y component
  eZ = ((errors == 2) | (errors == 3)).astype(int) # Y or Z component
  return eX, eZ

def syndrome(H, e):
  return (H @ e.T) % 2

Hx = np.random.randint(0,2,(10,20))
Hz = np.random.randint(0,2,(10,20))


'''
Usage:

# Load Hx and Hz
Hx = np.random.randint(0,2,(10,20))
Hz = np.random.randint(0,2,(10,20))

# Generate Pauli Errors
errors = generate_errors(500, 20, 0.34, 0.32, 0.34)

# Convert IXYZ to eX,eZ channels
eX, eZ = pauli_to_XZ(errors)

# Compute syndromes
sX = syndrome(Hx, eZ)
sZ = syndrome(Hz, eX)
'''
