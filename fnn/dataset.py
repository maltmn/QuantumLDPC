import numpy as np

from panqec.codes import surface_2d

# Select random Pauli errors on each qubit
def generate_errors(num_samples, n_qubits, pX, pY, pZ):
# PauliErrorModel(pX, pY, pZ)
  errors = np.random.choice(
    [0,1,2,3], # I,X,Y,Z : none, X, both, Z
    size = (num_samples, n_qubits),
    p = [1-pX-pY-pZ, pX, pY, pZ]
  )
  return errors
  #errors = [0, 2, 1, 0, 3]

# Convert errors to eX and eZ vectors
def pauli_to_XZ(errors):
  eX = ((errors == 1) | (errors == 2)).astype(int) # X or Y component
  eZ = ((errors == 2) | (errors == 3)).astype(int) # Y or Z component
  return eX, eZ
  #eX = [0, 1, 1, 0, 0]
  #eZ = [0, 1, 0, 0, 1]
  #13 entries for d=3

# compute syndrome = H matrix * e transpose (mod 2)
def syndrome(H, e):
  return (H @ e.T) % 2
  # sX = syndrome(Hz, eX) = [0, 0, 1]
  # sZ = syndrome(Hx, eZ) = [0, 1, 0]
  #6 entries for d=3
  #one syndrome -> many different e's

#d = L-1 = 3
code = surface_2d.RotatedPlanar2DCode(4)

#H matrices determine which qubits are checked for errors
#1: qubit included in syndrome calculation
#0: qubit not included in syndrome calculation

Hx = code.Hx.toarray()
Hz = code.Hz.toarray()

print(code.Hx.shape)
print(code.Hz.shape)

print("Dataset loaded from", __file__)
print("Lattice used", code.d)

#d=2, H = (4 rows, 9 columns) = (checks, qubits)
#d=3, H = (7 or 8 rows, 16 columns) 

'''
 Hx = np.array([
  [1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
  [1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
  [1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
  [0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
], dtype=np.int64)

Hz = np.array([
  [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
  [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
  [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
], dtype=np.int64)
'''

assert np.all((Hx @ Hz.T) % 2 == 0)

