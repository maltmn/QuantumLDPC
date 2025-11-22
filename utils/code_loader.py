import numpy as np

def load_code(Hx_path, Hz_path):
  Hx = np.load(Hx_path)
  Hz = np.load(Hz_path)
  return Hx,Hz
