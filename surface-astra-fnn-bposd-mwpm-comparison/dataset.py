import numpy as np

from panqec.codes import surface_2d

def generate_errors(num_samples, n_qubits, pX, pY, pZ):
# PauliErrorModel(pX, pY, pZ)
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

code = surface_2d.RotatedPlanar2DCode(3)

Hx = code.Hx.toarray()
Hz = code.Hz.toarray()

print(code.Hx.shape)
print(code.Hz.shape)

'''
 Hx = np.array([
  [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
  [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
  [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
  [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
  [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
  [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]
], dtype=np.int64)

Hz = np.array([
  [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
  [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
  [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
  [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
], dtype=np.int64)
'''

assert np.all((Hx @ Hz.T) % 2 == 0)

